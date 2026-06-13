#!/usr/bin/env python3
"""PE-Core image--text scoring over render tarballs.

Per-view score is the cosine similarity between normalized PE embeddings. The model and preprocessing come from
open_clip ("PE-Core-bigG-14-448", pretrained="meta") and run with fp32 weights under torch.autocast("cuda").

Render shard tar members are keyed `prompt_0000/prompt_0000_0000.png` or `<prompt_key>/<prompt_key>_0000.png`; the
captions JSONL line order defines prompt_index. Per prompt: score all 8 views, pool with max; the final metric is the
mean over prompts.
"""
import argparse
import io
import json
import re
import tarfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

MODEL_NAME = "PE-Core-bigG-14-448"
PRETRAINED = "meta"

_INDEX_MEMBER_RE = re.compile(
    r"^prompt_(?P<prompt_idx>\d+)/prompt_(?P=prompt_idx)_(?P<view_idx>\d+)\.(?:png|jpg|jpeg)$"
)
_KEY_MEMBER_RE = re.compile(r"^(?P<prompt_key>[^/]+)/(?P=prompt_key)_(?P<view_idx>\d+)\.(?:png|jpg|jpeg)$")


def load_captions(path: Path) -> list[dict]:
    records = []
    for i, line in enumerate(raw for raw in path.open() if raw.strip()):
        rec = json.loads(line)
        records.append(
            {"prompt_index": i, "prompt_key": rec.get("id", f"prompt_{i:04d}"), "caption": rec["caption"].strip()}
        )
    return records


def index_renders(images_dir: Path, prompts: list[dict]) -> dict[int, tuple[Path, list[str]]]:
    index_by_key = {p["prompt_key"]: p["prompt_index"] for p in prompts}
    renders = {}
    for tar_path in sorted(images_dir.glob("*.tar")):
        with tarfile.open(tar_path, "r") as tf:
            members = [m.name for m in tf.getmembers() if m.isfile()]
        by_prompt = defaultdict(list)
        for name in members:
            if match := _INDEX_MEMBER_RE.match(name):
                by_prompt[int(match.group("prompt_idx"))].append((int(match.group("view_idx")), name))
            elif (match := _KEY_MEMBER_RE.match(name)) and match.group("prompt_key") in index_by_key:
                by_prompt[index_by_key[match.group("prompt_key")]].append((int(match.group("view_idx")), name))
        for prompt_index, names in by_prompt.items():
            renders[prompt_index] = (tar_path, [n for _, n in sorted(names)])
    return renders


def join_records(prompts: list[dict], renders: dict) -> list[tuple[dict, Path, list[str]]]:
    """Pair each prompt with its render location."""
    missing = [prompt["prompt_key"] for prompt in prompts if prompt["prompt_index"] not in renders]
    if missing:
        raise FileNotFoundError(f"{len(missing)}/{len(prompts)} prompts have no rendered views (first: {missing[:5]})")
    records = []
    for prompt in prompts:
        tar_path, member_names = renders[prompt["prompt_index"]]
        records.append((prompt, tar_path, member_names))
    return records


class RenderDataset(Dataset):
    """Decodes and preprocesses one prompt's views; each worker keeps its own open tar handles."""

    def __init__(self, records, preprocess, tokenizer):
        self.records = records
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.handles = {}

    def __len__(self):
        return len(self.records)

    def _tar(self, path):
        if path not in self.handles:
            self.handles[path] = tarfile.open(path, "r")
        return self.handles[path]

    def __getitem__(self, i):
        prompt, tar_path, member_names = self.records[i]
        tf = self._tar(tar_path)
        images = [Image.open(io.BytesIO(tf.extractfile(n).read())).convert("RGB") for n in member_names]
        image_tensors = torch.stack([self.preprocess(img) for img in images])
        text_tokens = self.tokenizer([prompt["caption"]])
        return prompt, image_tensors, text_tokens


def collate(batch):
    prompts = [prompt for prompt, _, _ in batch]
    image_tensors = torch.cat([images for _, images, _ in batch])
    text_tokens = torch.cat([tokens for _, _, tokens in batch])
    n_views = [images.shape[0] for _, images, _ in batch]
    return prompts, image_tensors, text_tokens, n_views


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images-dir", required=True, help="Directory containing render shard tarballs")
    parser.add_argument("--captions-jsonl", required=True, help="JSONL with prompt captions, in prompt order")
    parser.add_argument("--out", required=True, help="Output JSONL")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers for decode and preprocessing")
    parser.add_argument("--batch-size", type=int, default=8, help="Prompts per forward (8 views each)")
    args = parser.parse_args()

    import open_clip

    prompts = load_captions(Path(args.captions_jsonl))
    with ThreadPoolExecutor(max_workers=1) as executor:
        indexing = executor.submit(index_renders, Path(args.images_dir), prompts)
        model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        model = model.cuda().eval()
        records = join_records(prompts, indexing.result())
    print(f"Evaluating {len(records)} prompts")
    loader = DataLoader(
        RenderDataset(records, preprocess, tokenizer),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    max_scores = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as out_f, torch.no_grad(), torch.autocast("cuda"):
        for batch_prompts, image_tensors, text_tokens, n_views in tqdm(loader):
            image_features = model.encode_image(image_tensors.cuda(), normalize=True)
            text_features = model.encode_text(text_tokens.cuda(), normalize=True)
            similarities = image_features @ text_features.T
            offset = 0
            for column, (prompt, n) in enumerate(zip(batch_prompts, n_views)):
                scores = similarities[offset : offset + n, column].cpu().tolist()
                offset += n
                row = {
                    "prompt_index": prompt["prompt_index"],
                    "prompt_key": prompt["prompt_key"],
                    "caption": prompt["caption"],
                    "scores": scores,
                    "max_score": max(scores),
                    "mean_score": sum(scores) / len(scores),
                }
                out_f.write(json.dumps(row) + "\n")
                max_scores.append(row["max_score"])

    print(f"Prompts scored: {len(max_scores)}")
    print(f"Mean of MAX scores: {sum(max_scores) / len(max_scores):.17f}")


if __name__ == "__main__":
    main()
