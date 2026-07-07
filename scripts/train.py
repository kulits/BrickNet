#!/usr/bin/env python3
"""Train on BrickNet (HF Transformers + PEFT).

Two modes:
  pt   pretraining: path + EOS, no loss masking.
  sft  caption conditioning on a merged PT adapter: caption + "\n" + path + EOS,
       loss on the path and EOS only.

Data loads from the jsonl.xz files; sequences are tokenized at runtime, and
training batches groups by length. Inference is done with scripts/generate.py.

Pretraining (paths_pt + paths_sft):
    python scripts/train.py pt \
        --model Qwen/Qwen3-0.6B \
        --paths paths_pt.jsonl.xz \
        --paths paths_sft.jsonl.xz \
        --val_paths paths_val.jsonl.xz \
        --output runs/pt

SFT (caption-walk pairs, on top of the PT adapter):
    python scripts/train.py sft \
        --model Qwen/Qwen3-0.6B \
        --pt_adapter kulits/BrickNet-0.6B-PT \
        --paths paths_sft.jsonl.xz \
        --captions captions_sft.jsonl.xz \
        --val_paths paths_val.jsonl.xz \
        --val_captions captions_val.jsonl.xz \
        --output runs/sft

Multi-GPU: launch with torchrun --nproc-per-node=N; the global batch size is fixed
by the recipe and split across processes.
"""

import argparse
import hashlib
import io
import json
import os
import subprocess
import tempfile
import time
from array import array
from pathlib import Path

import numpy as np
import torch
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss, apply_liger_kernel_to_qwen3
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AttentionInterface,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.trainer_pt_utils import LengthGroupedSampler

COMMON = dict(
    lora_r=32,
    lora_targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0,
    adam_betas=(0.9, 0.998),
    adam_eps=1e-5,
    weight_decay=0.033,
    clip_grad=1.0,
    min_lr_rate=0.01,
    save_interval=5000,
)
RECIPES = {
    "pt": dict(
        COMMON,
        max_len=6400,
        global_batch_size=32,
        train_iters=250_000,
        lr=5e-4,
        warmup_iters=12_500,
        lora_alpha=32.0,
        eval_interval=2500,
        val_samples=1000,
    ),
    "sft": dict(
        COMMON,
        max_len=2816,
        global_batch_size=8,
        train_iters=40_000,
        lr=2.5e-4,
        warmup_iters=2_000,
        lora_alpha=64.0,
        eval_interval=500,
    ),
}


def read_lines(path: str):
    if not path.endswith(".xz"):
        with open(path, "r", encoding="utf-8") as f:
            yield from f
        return
    # The xz binary decodes multi-block files in parallel (xz >= 5.4) and overlaps
    # decompression with parsing either way; Python's lzma is single-threaded.
    proc = subprocess.Popen(["xz", "-dcT0", path], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    with io.TextIOWrapper(proc.stdout, encoding="utf-8") as f:
        yield from f
    if proc.wait() != 0:
        raise RuntimeError(f"xz failed on {path}")


def read_jsonl(path: str):
    return map(json.loads, read_lines(path))


def first_round_rows(paths_file: str):
    """Rows sharing the first row's sampling round; the round tag substring is a
    prefilter so only candidate rows pay for JSON parsing. The shipped data
    includes multiple sampling rounds, though only the first was used."""
    head = next(read_jsonl(paths_file), None)
    if head is None:
        raise SystemExit(f"empty paths file: {paths_file}")
    tag = f'"round": {json.dumps(head["round"])}'
    for line in read_lines(paths_file):
        if tag in line:
            row = json.loads(line)
            if row["round"] == head["round"]:
                yield row


def pt_paths(paths_files):
    return (row["path"].strip(" ") for pf in paths_files for row in first_round_rows(pf))


def sft_pairs(paths_file, captions_file):
    """One (caption, path) pair per first-round walk: the walk with sample_index k
    pairs with its model's k-th caption. Captions whose walk is absent are skipped."""
    captions = {}
    for row in read_jsonl(captions_file):
        captions.setdefault(str(row["id"]), []).append(row["caption"].strip(" "))
    pairs = []
    for row in first_round_rows(paths_file):
        caps = captions.get(str(row["source"]))
        k = int(row["sample_index"])
        if caps and k < len(caps):
            pairs.append((caps[k], row["path"].strip(" ")))
    return pairs


class TextDataset(torch.utils.data.Dataset):
    """Items are path strings (pt) or (caption, path) tuples (sft), held in a
    memmapped scratch file so ranks and dataloader workers share page cache
    instead of anonymous RAM; runs finding an existing scratch skip the corpus
    parse. Rank 0 builds; other ranks wait for the marker. `lengths` is a
    character-count proxy for length-grouped batching."""

    def __init__(self, blob_path, offsets, pair):
        self.blob = np.memmap(blob_path, dtype=np.uint8, mode="r")
        self.offsets = offsets
        self.pair = pair
        d = np.diff(offsets)
        self.lengths = (d[0::2] + d[1::2] if pair else d).tolist()

    @classmethod
    def build(cls, make_items, pair, scratch, name):
        blob_path = scratch / f"{name}.bin"
        idx_path = scratch / f"{name}.idx.npy"
        done = scratch / f"{name}.done"
        if not done.exists():
            if int(os.environ.get("RANK", "0")) == 0:
                scratch.mkdir(parents=True, exist_ok=True)
                offsets = array("q", [0])
                pos = 0
                with open(blob_path, "wb") as f:
                    for item in make_items():
                        for part in item if pair else (item,):
                            b = part.encode()
                            f.write(b)
                            pos += len(b)
                            offsets.append(pos)
                if pos == 0:
                    raise SystemExit(f"no items for {name}")
                np.save(idx_path, np.frombuffer(offsets, dtype=np.int64))
                done.touch()
            else:
                while not done.exists():
                    time.sleep(5)
        return cls(blob_path, np.load(idx_path), pair)

    def _get(self, j):
        return self.blob[self.offsets[j] : self.offsets[j + 1]].tobytes().decode()

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, i):
        return (self._get(2 * i), self._get(2 * i + 1)) if self.pair else self._get(i)


class Collator:
    def __init__(self, tokenizer, max_len):
        self.tok = tokenizer
        self.max_len = max_len
        self.eos = tokenizer.eos_token_id
        self.nl = tokenizer("\n", add_special_tokens=False)["input_ids"][0]

    def _encode(self, texts):
        return self.tok(texts, add_special_tokens=False)["input_ids"]

    def __call__(self, batch):
        if isinstance(batch[0], str):
            seqs = [ids[: self.max_len] + [self.eos] for ids in self._encode(batch)]
            label_starts = [0] * len(seqs)
        else:
            caps = self._encode([c for c, _ in batch])
            paths = self._encode([p for _, p in batch])
            seqs, label_starts = [], []
            for c, p in zip(caps, paths):
                budget = max(self.max_len - len(c) - 1, 1)
                seqs.append(c + [self.nl] + p[:budget] + [self.eos])
                label_starts.append(len(c) + 1)
        # Right padding needs no attention mask: pads sit in the causal future of
        # every real token, and their labels are -100.
        width = (max(len(s) for s in seqs) + 63) // 64 * 64
        n = len(seqs)
        input_ids = np.full((n, width), self.eos, dtype=np.int64)
        labels = np.full((n, width), -100, dtype=np.int64)
        for r, (s, a) in enumerate(zip(seqs, label_starts)):
            input_ids[r, : len(s)] = s
            labels[r, a : len(s)] = s[a:]
        return {
            "input_ids": torch.from_numpy(input_ids),
            "labels": torch.from_numpy(labels),
        }


class LoraTrainer(Trainer):
    """Per-rank token-mean masked cross entropy, computed with Liger's fused kernel
    so the full logits are never materialized; batches grouped by similar length to
    bound padding."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._flce = LigerFusedLinearCrossEntropyLoss(reduction="mean")

    def _get_train_sampler(self, *a, **kw):
        return LengthGroupedSampler(self.args.train_batch_size, lengths=self.train_dataset.lengths)

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs["labels"]
        outputs = model(
            input_ids=inputs["input_ids"],
            output_hidden_states=True,
            logits_to_keep=1,
        )
        lm_head = self.model.get_base_model().lm_head
        hidden = outputs.hidden_states[-1].to(lm_head.weight.dtype)
        shift_labels = torch.cat([labels[:, 1:], labels.new_full((labels.shape[0], 1), -100)], dim=1)
        loss = self._flce(lm_head.weight, hidden.reshape(-1, hidden.shape[-1]), shift_labels.reshape(-1))
        return (loss, None) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss, None, None)


def sdpa_causal(module, query, key, value, attention_mask, **kwargs):
    # Maskless SDPA dispatches to the fused causal kernel; the built-in "sdpa"
    # implementation materializes a dense mask under torch.compile
    # (https://github.com/huggingface/transformers/issues/39017).
    return sdpa_attention_forward(module, query, key, value, None, **kwargs)


def load_model(model_id, adapters, recipe):
    AttentionInterface.register("sdpa_causal", sdpa_causal)
    apply_liger_kernel_to_qwen3()

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation="sdpa_causal")
    for adapter in adapters or []:
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    cfg = LoraConfig(
        r=recipe["lora_r"],
        lora_alpha=recipe["lora_alpha"],
        lora_dropout=recipe["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=recipe["lora_targets"],
    )
    model = get_peft_model(model, cfg)
    model.config.use_cache = False
    return model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["pt", "sft"])
    p.add_argument("--model", required=True, help="Base HF model (path or hub id).")
    p.add_argument(
        "--pt_adapter",
        action="append",
        default=None,
        help="LoRA adapter(s) merged into the base before training (sft mode).",
    )
    p.add_argument("--paths", action="append", required=True, help="paths jsonl.xz (repeatable; pt combines them).")
    p.add_argument("--captions", default=None, help="captions_sft.jsonl.xz (sft mode).")
    p.add_argument("--val_paths", default=None)
    p.add_argument("--val_captions", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--scratch", default=None, help="Local scratch dir for the memmapped corpus (default: system temp).")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    if args.mode == "sft":
        if not (args.pt_adapter and args.captions):
            p.error("sft mode requires --pt_adapter and --captions")
        if len(args.paths) > 1:
            p.error("sft mode takes a single --paths file")
        if args.val_paths and not args.val_captions:
            p.error("sft mode requires --val_captions with --val_paths")

    # Tokenization happens inside dataloader workers; the main process must not
    # spin up tokenizer thread pools before the workers fork.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    recipe = RECIPES[args.mode]

    world = int(os.environ.get("WORLD_SIZE", "1"))
    if recipe["global_batch_size"] % world:
        raise SystemExit(f"global batch size {recipe['global_batch_size']} not divisible by {world} processes")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    sources = [args.mode] + [
        str(Path(f).resolve()) for f in [*args.paths, args.captions, args.val_paths, args.val_captions] if f
    ]
    ident = hashlib.md5(json.dumps(sources).encode()).hexdigest()[:10]
    scratch = Path(args.scratch or tempfile.gettempdir()) / f"bricknet_{args.mode}_{ident}"

    val_ds = None
    if args.mode == "pt":
        train_ds = TextDataset.build(lambda: pt_paths(args.paths), False, scratch, "train")
        if args.val_paths:

            def val_items():
                rows = list(pt_paths([args.val_paths]))
                pick = np.random.default_rng(0).choice(
                    len(rows), size=min(recipe["val_samples"], len(rows)), replace=False
                )
                return [rows[i] for i in pick]

            val_ds = TextDataset.build(val_items, False, scratch, "val")
    else:
        train_ds = TextDataset.build(lambda: sft_pairs(args.paths[0], args.captions), True, scratch, "train")
        if args.val_paths:
            val_ds = TextDataset.build(lambda: sft_pairs(args.val_paths, args.val_captions), True, scratch, "val")
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"train sequences: {len(train_ds):,}")

    model = load_model(args.model, args.pt_adapter, recipe)

    targs = TrainingArguments(
        output_dir=args.output,
        max_steps=recipe["train_iters"],
        per_device_train_batch_size=recipe["global_batch_size"] // world,
        per_device_eval_batch_size=recipe["global_batch_size"] // world,
        learning_rate=recipe["lr"],
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": recipe["min_lr_rate"]},
        warmup_steps=recipe["warmup_iters"],
        weight_decay=recipe["weight_decay"],
        adam_beta1=recipe["adam_betas"][0],
        adam_beta2=recipe["adam_betas"][1],
        adam_epsilon=recipe["adam_eps"],
        max_grad_norm=recipe["clip_grad"],
        bf16=True,
        logging_steps=100,
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=recipe["eval_interval"],
        save_steps=recipe["save_interval"],
        save_total_limit=3,
        dataloader_num_workers=2,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to=[],
        torch_compile=True,
    )

    trainer = LoraTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=Collator(tokenizer, recipe["max_len"]),
    )
    trainer.train(resume_from_checkpoint=args.resume and any(Path(args.output).glob("checkpoint-*")))

    if trainer.is_world_process_zero():
        final = Path(args.output) / "adapter"
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.data = param.data.to(torch.bfloat16)
        model.save_pretrained(final)
        print(f"saved adapter to {final}")


if __name__ == "__main__":
    main()
