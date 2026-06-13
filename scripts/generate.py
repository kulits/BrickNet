#!/usr/bin/env python3
r"""Generate LEGO structures with a fine-tuned model.

Supports two model loading modes:
  --model <path>                       Full merged HF checkpoint.
  --model <base> --lora <a1> --lora <a2>  Base HF model + LoRA adapter(s), merged in order.

Unconditional (PT):
    python scripts/generate.py --model Qwen/Qwen3-0.6B --lora kulits/BrickNet-0.6B-PT \
        --output out.jsonl --num_samples 2048 --batch_size 128 --stop_after_newlines 199

Conditional (SFT):
    python scripts/generate.py --model Qwen/Qwen3-0.6B \
        --lora kulits/BrickNet-0.6B-PT --lora kulits/BrickNet-0.6B-SFT \
        --output out.jsonl --prompts_file prompts.jsonl --batch_size 128

Multi-GPU (accelerate splits samples across GPUs, each loads the full model):
    accelerate launch --num_processes=N scripts/generate.py --model Qwen/Qwen3-0.6B \
        --lora kulits/BrickNet-0.6B-PT --output out.jsonl --num_samples 4096
"""

import argparse
import json
import os

import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList


class StopAfterNewlines:
    def __init__(self, newline_id: int, target: int, eos_id: int, prompt_len: int):
        self.newline_id = newline_id
        self.target = target
        self.eos_id = eos_id
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores):
        gen = input_ids[:, self.prompt_len :]
        nl_counts = (gen == self.newline_id).sum(dim=1)
        done = nl_counts >= self.target
        scores[~done, self.eos_id] = float("-inf")
        scores[done] = float("-inf")
        scores[done, self.eos_id] = 0.0
        return scores


def _load_prompts(path: str) -> list[str]:
    with open(path) as f:
        captions = [json.loads(line)["caption"] for line in f]
    assert captions, f"no prompts found in {path}"
    return captions


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model path (base model when --lora is given).")
    p.add_argument("--lora", action="append", default=None, help="PEFT LoRA adapter path(s), applied in order.")
    p.add_argument("--output", required=True)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--dtype", default="bfloat16", help="Model dtype (e.g. bfloat16, float16, float32).")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--prompt", default="a", help="Fixed prompt for unconditional generation (default: 'a').")
    mode.add_argument("--prompts_file", default=None, help="JSONL with per-sample captions (field: 'caption').")
    p.add_argument("--num_samples", type=int, default=100, help="Number of samples (unconditional mode only).")
    p.add_argument("--n_per_prompt", type=int, default=1, help="Completions per caption (prompts_file mode only).")
    p.add_argument("--seed", type=int, default=None, help="Base random seed (default: unseeded).")
    p.add_argument("--stop_after_newlines", type=int, default=None, help="Force EOS after N newlines.")
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True

    accelerator = Accelerator()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": accelerator.local_process_index},
        dtype=args.dtype,
    )
    if args.lora:
        from peft import PeftModel

        for lora_path in args.lora:
            model = PeftModel.from_pretrained(model, lora_path)
            model = model.merge_and_unload()
    model.eval()

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tok.pad_token = tok.eos_token
    model.generation_config.pad_token_id = tok.pad_token_id
    nl_id = tok.encode("\n", add_special_tokens=False)[0]
    pad_id = tok.pad_token_id

    if args.prompts_file:
        captions = _load_prompts(args.prompts_file)
        work = []
        for i, c in enumerate(captions):
            caption_ids = tok.encode(c, add_special_tokens=False) + [nl_id]
            for k in range(args.n_per_prompt):
                work.append((i, k, caption_ids))
    else:
        prompt_ids = tok.encode(args.prompt, add_special_tokens=False)
        assert prompt_ids, f"prompt {args.prompt!r} encodes to zero tokens"
        work = [(i, 0, prompt_ids) for i in range(args.num_samples)]

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    shard_path = f"{args.output}.shard{accelerator.process_index}"
    with accelerator.split_between_processes(work) as local_work, open(shard_path, "w") as f:
        for batch_start in range(0, len(local_work), args.batch_size):
            if args.seed is not None:
                torch.manual_seed(args.seed + accelerator.process_index * 1000000 + batch_start)
            batch = local_work[batch_start : batch_start + args.batch_size]
            token_lists = [w[2] for w in batch]

            max_len = max(len(t) for t in token_lists)
            padded = [[pad_id] * (max_len - len(t)) + t for t in token_lists]
            inp = torch.tensor(padded, device=accelerator.device)
            attn = (inp != pad_id).long()

            processors = None
            if args.stop_after_newlines is not None:
                processors = LogitsProcessorList(
                    [StopAfterNewlines(nl_id, args.stop_after_newlines, tok.eos_token_id, max_len)]
                )

            with torch.inference_mode():
                out = model.generate(inp, attention_mask=attn, logits_processor=processors, **gen_kwargs)

            for i, (gid, sample, toks) in enumerate(batch):
                generated = tok.decode(out[i][max_len:], skip_special_tokens=True)
                if args.prompts_file:
                    text = generated
                else:
                    text = args.prompt + generated
                f.write(json.dumps({"id": gid, "sample": sample, "text": text}) + "\n")
            f.flush()
            print(f"GPU{accelerator.local_process_index}: {batch_start + len(batch)}/{len(local_work)}", flush=True)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        records = []
        for i in range(accelerator.num_processes):
            shard = f"{args.output}.shard{i}"
            if os.path.exists(shard):
                with open(shard) as f:
                    records.extend(json.loads(line) for line in f)
                os.remove(shard)
        records.sort(key=lambda r: (r["id"], r["sample"]))
        with open(args.output, "w") as out:
            for r in records:
                out.write(json.dumps(r) + "\n")
        print(f"Done. {len(records)}/{len(work)} samples written to {args.output}")


if __name__ == "__main__":
    main()
