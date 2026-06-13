"""Score generated path-text samples: parse validity and collisions.

Input: jsonl whose rows carry the sample text under "path" (or "text"). Output rows pass through
with three fields (re)written:
  n_actions  -- actions attempted in the text (one part per action)
  invalid    -- index of the first unparsable action, or null
  collisions -- indices of actions whose part collides under autoregressive placement
                (null with --no-collision, which needs no meshes)

Usage: python -m bricknet score in.jsonl out.jsonl [--workers N] [--no-collision]
"""

import argparse
import functools
import json
import os
from concurrent.futures import ProcessPoolExecutor

from .collision import check_placements, first_collision
from .core import FixedEdge, Tree
from .graph import decode_graph, tree_to_graph
from .tree import parse_sample


def _fixed_parents(tree: Tree) -> dict[int, int]:
    """child part index -> parent part index for fixed edges (rigid mounts may interpenetrate)."""
    return {i + 1: e.parent for i, e in enumerate(tree.edges) if isinstance(e, FixedEdge)}


def check_tree(tree: Tree) -> list[int]:
    """Part indices that collide under autoregressive placement (collision.check_placements)."""
    return check_placements([p.part_id for p in tree.parts], decode_graph(tree_to_graph(tree)), _fixed_parents(tree))


def collision_free_prefix(tree: Tree) -> int:
    """Length of the longest collision-free placement prefix (len(tree.parts) when clean)."""
    return first_collision([p.part_id for p in tree.parts], decode_graph(tree_to_graph(tree)), _fixed_parents(tree))


def score_text(text: str, collision: bool = True) -> tuple[int, int | None, list[int] | None]:
    if not text.endswith("\n"):
        text += "\n"
    n_actions = (text.count("\n") + 2) // 2
    res = parse_sample(text)
    invalid = None if res.error is None else len(res.tree.parts)
    return n_actions, invalid, check_tree(res.tree) if collision else None


def _score_rows(rows: list[dict], collision: bool = True) -> list[dict]:
    for row in rows:
        n_actions, invalid, collisions = score_text(row.get("path") or row["text"], collision)
        row["n_actions"], row["invalid"], row["collisions"] = n_actions, invalid, collisions
    return rows


def score_file(in_path: str, out_path: str, workers: int = 1, collision: bool = True) -> list[dict]:
    rows = [json.loads(ln) for ln in open(in_path)]
    if workers > 1:
        os.environ["TBB_NUM_THREADS"] = "1"  # meshlib TBB pools thrash under multiprocessing
        chunks = [rows[i::workers] for i in range(workers)]
        with ProcessPoolExecutor(workers) as ex:
            scored = list(ex.map(functools.partial(_score_rows, collision=collision), chunks))
        rows = [None] * sum(map(len, scored))
        for i, chunk in enumerate(scored):  # invert the rows[i::workers] striping so output order matches input order
            rows[i::workers] = chunk
    else:
        rows = _score_rows(rows, collision)
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return rows


def summarize(rows: list[dict]) -> str:
    n = len(rows)
    full = sum(r["invalid"] is None for r in rows)
    out = f"{n} samples | fully parsable: {full} ({full / n:.1%})"
    if all(r["collisions"] is not None for r in rows):
        clean = sum(r["invalid"] is None and not r["collisions"] for r in rows)
        first_bad = [min([r["n_actions"] if r["invalid"] is None else r["invalid"]] + r["collisions"]) for r in rows]
        out += (
            f" | parsable and collision-free: {clean} ({clean / n:.1%})"
            f" | mean actions before first failure: {sum(first_bad) / n:.1f}"
        )
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="bricknet score", description="Score generated path text for parsability and collisions."
    )
    ap.add_argument("input", help="jsonl of generated samples")
    ap.add_argument("output", help="jsonl of per-sample scores")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-collision", action="store_true", help="parse metrics only; needs no meshes")
    args = ap.parse_args(argv)
    print(summarize(score_file(args.input, args.output, args.workers, collision=not args.no_collision)))


if __name__ == "__main__":
    main()
