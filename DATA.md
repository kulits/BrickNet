# Bundled data (`bricknet/_data/v1`)

These ship inside the package; `BRICKNET_CATALOG` (default `v1`) selects the catalog version.

- `part_names.json` — the part vocabulary, `stem -> name`; `part_id` = file-order index.
- `car.ldr` — an example model (at `_data/`, outside the versioned catalog).
- `color_names.json` — color name -> LDraw color code, including extended codes
  that are not in the standard `LDConfig.ldr`.
- `labels.json.xz` — per-part connector labels: `{stem.dat: {kind: {subtype: rows}}}`, with one more
  level for the polarized kinds (`hinge`/`ball`/`fixed`: `{subtype: {"in"|"on": rows}}`). A row is
  `[x, y, z, pitch_deg, roll_deg, ...]` in part-local LDU; a sixth column is the frame yaw for `fixed`
  connectors and the span length for `axle` connectors, and is absent otherwise. The mating axis of a
  connector frame is its -Y. Iterating `sorted(kind) -> sorted(subtype) -> sorted(polarity) -> row order`
  defines the **canonical connector order**: the flat indices used by the graphs' `edge_idx` and, counted
  within one (subtype, polarity) subgroup, the connector letters used by path text.
- `part_aliases.json.xz` — stem canonicalization for LDraw part references: each row maps an obsolete or
  duplicate part stem (`src`) to a canonical stem (`dst`). The mapping covers official LDraw `~Moved to`
  and `!LDRAW_ORG Part Alias` / `Shortcut Alias` wrapper files (chains resolved to their final target) plus
  parts whose header title and geometry are identical (folded to the lexicographically smallest stem).
  `final_matrix_3x4` is the LDraw line-type-1 transform `(x y z a b c d e f g h i)` of the wrapper, composed
  into the referencing transform on substitution (`T_new = T_part @ T_alias`); identity rows are pure
  renames. The vocabulary contains no `src` stem; references resolve through this mapping before lookup.

# Large files (direct download)

- `inset.tar.xz` (369 MB) — per-part collision meshes (watertight PLYs, inset inward 0.25 LDU, Z negated).
  `python -m bricknet fetch-meshes` downloads and extracts them into `<data dir>/inset/` (one `.ply` per
  part stem), where the data dir is `$BRICKNET_DATA` if set, else the platform user-data directory.
  URL: <https://keeper.mpdl.mpg.de/seafhttp/f/77b868746e6d41baa791/?op=view>
- `ldraw.tar.xz` (98 MB) — the LDraw part library snapshot the meshes and vocabulary were built from
  (`ldraw/` at the archive root; covers every `part_names.json` stem).
  URL: <https://keeper.mpdl.mpg.de/seafhttp/f/9b4a3c04e88940d1a718/?op=view>

# Datasets (request form)

The datasets are gated; request access at <https://forms.gle/dm4eYSa5gh4DqzRT6>. They are not needed to
use the library. Three splits: **pt** (pretraining), **sft** (caption-conditioned fine-tuning), and
**val** (the 512 evaluation models). The splits are disjoint at the model level; every file below carries the
splits' model sets exactly.

## Graphs — `pt.npz.xz`, `sft.npz.xz`, `val.npz.xz`

Connectivity graphs (one per model), CSR-batched. Decompress, then read with `bricknet.load_graphs`,
which returns `Graph` objects (no absolute transforms; realize one with `bricknet.decode_graph` or
`bricknet.graph_to_ldr`). Arrays inside the npz:

| array | shape | contents |
| --- | --- | --- |
| `part_ids` / `colors` | (total parts,) | vocabulary index / color code per part |
| `node_ptr` / `edge_ptr` | (graphs + 1,) | CSR offsets: graph `i` owns parts `node_ptr[i]:node_ptr[i+1]` |
| `edge_kind` | (total edges,) | joint family (stud / hinge / axle / ball / fixed) |
| `edge_idx` | (total edges, 4) | `[a, a_conn, b, b_conn]` — part indices (graph-local) and canonical flat connector indices |
| `edge_yaw` | (total edges,) | stud/hinge: yaw in radians; **axle: slide in LDU**; ball/fixed: 0 |
| `edge_flip` | (total edges,) | hinge/axle anti-parallel mating; false otherwise |
| `edge_rot` | (total edges, 3) | ball: xyz Euler angles in radians; axle: `[rotation_rad, 0, 0]`; zeros otherwise |

Edge sides are canonical: `a` is the stud (vs. hole), the `"in"` polarity (hinge/ball/fixed), or the
axle-family socket/cross/clip side.

| split | graphs | parts | edges |
| --- | --- | --- | --- |
| pt | 253,623 | 38,775,582 | 125,816,010 |
| sft | 67,185 | 1,774,387 | 4,124,165 |
| val | 512 | 13,397 | 32,547 |

## Captions — `captions_sft.jsonl.xz`, `captions_val.jsonl.xz`

One JSON object per line: `{"id": <model id>, "caption": <text>}`. Line order follows npz row order:
the sft file has five captions per model (lines `5i..5i+4` describe sft.npz row `i`); the val file has one
(line `i` describes val.npz row `i`). The `id` joins to the path files' `source` (minus the `.npz` suffix).

## Sampled paths — `paths_pt.jsonl.xz`, `paths_sft.jsonl.xz`, `paths_val.jsonl.xz`

Collision-free build sequences sampled from the graphs: five independent sampling rounds per model, more
samples for larger components, deduplicated by exact path text within each split, shuffled. One JSON object
per line:

| field | contents |
| --- | --- |
| `source` | model id (`<id>.npz`) the walk was sampled from |
| `round` | sampling round |
| `sample_index` | index of the walk within its (round, model) |
| `component_index` / `component_nodes` | which connected component, and its size |
| `nodes` | parts placed by this walk |
| `complete_component` / `complete_npz` | whether the walk covered the component / the whole model |
| `path` | the build sequence in path text, the format `bricknet.parse_sample` reads (full grammar: the `bricknet.tree` module docstring) |

| split | rows |
| --- | --- |
| pt | 32,572,697 |
| sft | 5,912,934 |
| val | 45,516 |

## Training recipe

The **PT** models were trained on the pt and sft path pools combined and deduplicated by path text
(8,092,423 sequences), tokenized with the Qwen3 tokenizer and packed to 6,400-token sequences, loss
over the full sequence. The **SFT** models were trained on one (caption, path) pair per caption —
five per model — tokenized jointly as `caption + "\n" + path` and packed to 2,816-token sequences
with loss on the path tokens only; the held-out val captions serve as evaluation prompts.
