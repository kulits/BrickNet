#!/usr/bin/env python3
"""VQAScore over render tarballs with Qwen2-VL.

Per-view score is P("Yes") for the first generated token given the question

    Is this LEGO set {caption, lowercased, terminal punctuation stripped}?

Render shard tar members are keyed `prompt_0000/prompt_0000_0000.png` or `<prompt_key>/<prompt_key>_0000.png`; the
captions JSONL line order defines prompt_index. Per prompt: score all 8 views, pool with max; the final metric is the
mean over prompts.
"""
import argparse
import io
import json
import os
import re
import tarfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

_INDEX_MEMBER_RE = re.compile(
    r"^prompt_(?P<prompt_idx>\d+)/prompt_(?P=prompt_idx)_(?P<view_idx>\d+)\.(?:png|jpg|jpeg)$"
)
_KEY_MEMBER_RE = re.compile(r"^(?P<prompt_key>[^/]+)/(?P=prompt_key)_(?P<view_idx>\d+)\.(?:png|jpg|jpeg)$")
_TERMINAL_PUNCT_RE = re.compile(r"[.!?]+$")


def build_question(caption: str) -> str:
    base = " ".join(caption.split()).strip()
    return f"Is this LEGO set {_TERMINAL_PUNCT_RE.sub('', base).lower()}?"


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
    """Builds one prompt's processed VQA inputs; each worker keeps its own open tar handles."""

    def __init__(self, records, processor):
        self.records = records
        self.processor = processor
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
        question = build_question(prompt["caption"])
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            for image in images
        ]
        texts = [
            self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            for message in messages
        ]
        vision_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=vision_inputs[0],
            videos=vision_inputs[1],
            padding=True,
            return_tensors="pt",
        )
        return prompt, question, inputs


def collate_single(batch):
    return batch[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images-dir", required=True, help="Directory containing render shard tarballs")
    parser.add_argument("--captions-jsonl", required=True, help="JSONL with prompt captions, in prompt order")
    parser.add_argument("--out", required=True, help="Output JSONL")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers for decode and preprocessing")
    parser.add_argument("--attn-implementation", default="sdpa", help="attention backend passed to from_pretrained")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    prompts = load_captions(Path(args.captions_jsonl))
    with ThreadPoolExecutor(max_workers=1) as executor:
        indexing = executor.submit(index_renders, Path(args.images_dir), prompts)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
            device_map="auto",
        ).eval()
        processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)
        yes_token_id = processor.tokenizer.encode("Yes")[0]
        records = join_records(prompts, indexing.result())
    print(f"Evaluating {len(records)} prompts")
    loader = DataLoader(
        RenderDataset(records, processor),
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=collate_single,
    )

    max_scores = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as out_f, torch.no_grad():
        for prompt, question, inputs in tqdm(loader):
            inputs = inputs.to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
            probs = torch.nn.functional.softmax(outputs.scores[0], dim=-1)
            scores = probs[:, yes_token_id].float().cpu().tolist()
            row = {
                "prompt_index": prompt["prompt_index"],
                "prompt_key": prompt["prompt_key"],
                "caption": prompt["caption"],
                "question": question,
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
