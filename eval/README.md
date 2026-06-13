# Render evaluation (PE / SigLIP 2 / VQAScore)

The image--text metrics (Table 3). Each metric is a self-contained script that loads only its
own model; all models come from pip packages and the HuggingFace hub. These scripts are part of the broader repository, not
of the `bricknet` package, and bring their own dependencies:
`torch transformers open_clip_torch qwen-vl-utils pillow tqdm`.

## Protocol

Inputs are webdataset-style shard tarballs of renders (8 views per prompt, members named
`prompt_0000/prompt_0000_0000.png` or `<prompt_key>/<prompt_key>_0000.png`) plus a captions JSONL whose line order
defines `prompt_index` (fields: `id`, `caption`).

For every prompt: score all 8 views independently, pool with **max**, report the **mean over prompts** of the pooled
score.

| Script | Model | Per-view score | Text input |
| --- | --- | --- | --- |
| `pe.py` | `PE-Core-bigG-14-448` via open_clip (`pretrained="meta"`) | cosine of normalized embeddings | caption verbatim |
| `siglip2.py` | `google/siglip2-giant-opt-patch16-384` | `sigmoid(logits_per_image)` | caption verbatim, `max_length=64` |
| `vqascore.py` | `Qwen/Qwen2-VL-7B-Instruct` | P("Yes") of first generated token | `Is this LEGO set {caption.lower(), no terminal punct}?` |

## Usage

```bash
export HF_HOME=...   # models resolve through the standard HF cache; offline works if cached
python eval/pe.py       --images-dir <render_shards> --captions-jsonl <prompts.jsonl> --out pe.jsonl
python eval/siglip2.py  --images-dir <render_shards> --captions-jsonl <prompts.jsonl> --out siglip2.jsonl
python eval/vqascore.py --images-dir <render_shards> --captions-jsonl <prompts.jsonl> --out vqa.jsonl
```

Each script takes `--images-dir`, `--captions-jsonl`, `--out`, and `--num-workers`; PE and SigLIP 2 add
`--batch-size`, VQAScore adds `--attn-implementation`. Output rows are
`{"prompt_index", "prompt_key", "caption", "scores", "max_score", "mean_score"}`; the reported metric is
the printed mean of `max_score` over prompts.
Every prompt in the captions JSONL must have rendered views; missing renders are an error. For a smoke test or
subset, pass a prefix of the captions JSONL (line order defines `prompt_index`, so only prefixes stay aligned for
index-keyed tar members; key-keyed members align regardless).
