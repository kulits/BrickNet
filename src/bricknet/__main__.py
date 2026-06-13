"""Command-line interface: python -m bricknet <command>."""

import argparse
import sys
from pathlib import Path

INSET_URL = "https://keeper.mpdl.mpg.de/seafhttp/f/77b868746e6d41baa791/?op=view"


def _read(path):
    return sys.stdin.read() if path == "-" else Path(path).read_text()


def _write(text, out):
    if out is None or out == "-":
        sys.stdout.write(text)
    else:
        Path(out).write_text(text)


def _convert_many(in_dir, out_dir, pattern, suffix, convert):
    """Convert every `pattern` file under in_dir into out_dir, skipping failures."""
    if out_dir is None:
        sys.exit("-o OUTPUT_DIR is required when the input is a directory")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = n_bad = 0
    for p in sorted(Path(in_dir).glob(pattern)):
        try:
            text = convert(p.read_text())
        except Exception as e:
            print(f"skip {p.name}: {e}", file=sys.stderr)
            n_bad += 1
            continue
        (out_dir / (p.stem + suffix)).write_text(text)
        n_ok += 1
    print(f"{n_ok} converted, {n_bad} skipped -> {out_dir}", file=sys.stderr)


def _cmd_path2ldr(args):
    import json

    from .graph import graph_to_ldr, tree_to_graph
    from .tree import parse_sample

    def convert(text):
        res = parse_sample(text)
        if res.error is not None:
            raise ValueError(f"parse error: {res.error}")
        return graph_to_ldr(tree_to_graph(res.tree))

    if args.path != "-" and Path(args.path).is_dir():
        _convert_many(args.path, args.output, "*.txt", ".ldr", convert)
    elif args.path.endswith(".jsonl"):
        # generator output: one .ldr per row, named by its id (or line number)
        if args.output is None:
            sys.exit("-o OUTPUT_DIR is required for .jsonl input")
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        n_ok = n_bad = 0
        for i, line in enumerate(open(args.path)):
            row = json.loads(line)
            name = str(row.get("id", i))
            try:
                ldr = convert(row.get("path") or row["text"])
            except (KeyError, ValueError) as e:
                print(f"skip {name}: {e}", file=sys.stderr)
                n_bad += 1
                continue
            (out_dir / f"{name}.ldr").write_text(ldr)
            n_ok += 1
        print(f"{n_ok} converted, {n_bad} skipped -> {out_dir}", file=sys.stderr)
    else:
        try:
            out = convert(_read(args.path))
        except ValueError as e:
            sys.exit(str(e))
        _write(out, args.output)


def _cmd_sample(args):
    import json

    from . import graph
    from .tree import serialize_tree

    src = Path(args.input)
    if src.suffix == ".npz":
        models = [(f"{src.stem}[{i}]", g) for i, g in enumerate(graph.load_graphs(src))]
    else:
        paths = sorted(src.glob("*.ldr")) if src.is_dir() else [src]
        models = []
        for p in paths:
            try:
                models.append((p.name, graph.parse_ldr(p.read_text())))
            except KeyError as e:
                print(f"skip {p.name}: unknown part {e}", file=sys.stderr)

    out = sys.stdout if args.output in (None, "-") else open(args.output, "w")
    n_rows = 0
    for source, g in models:
        total = len(g.part_ids)
        for ci, comp in enumerate(g.components):
            if len(comp) < 2:
                continue
            for k in range(args.n):
                seed = None if args.seed is None else args.seed + k
                t = graph.sample_collision_free_tree(g, ci, seed=seed)
                row = {
                    "source": source,
                    "sample_index": k,
                    "component_index": ci,
                    "component_nodes": len(comp),
                    "nodes": len(t.parts),
                    "complete_component": len(t.parts) == len(comp),
                    "complete_npz": len(t.parts) == total,
                    "path": serialize_tree(t),
                }
                out.write(json.dumps(row) + "\n")
                n_rows += 1
    if out is not sys.stdout:
        out.close()
    print(f"{n_rows} paths from {len(models)} models", file=sys.stderr)


def _cmd_score(args):
    from .score import main

    main(args.rest)


def _cmd_fetch_meshes(args):
    import io
    import lzma
    import tarfile
    import urllib.request

    from .collision import data_dir

    dest = Path(args.dest) if args.dest else data_dir() / "inset"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"downloading inset meshes (369 MB) -> {dest}")
    with urllib.request.urlopen(INSET_URL) as r:
        buf = io.BytesIO(r.read())
    with tarfile.open(fileobj=lzma.open(buf)) as tar:
        tar.extractall(dest, filter="data")
    n = len(list(dest.glob("*.ply")))
    print(f"extracted {n} meshes")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bricknet", description=__doc__)
    from . import __version__

    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("path2ldr", help="path text -> LDR model(s)")
    p.add_argument("path", help="path-text file, .jsonl of generated samples, directory of .txt files, or - for stdin")
    p.add_argument("-o", "--output", help="output file or directory (default stdout)")
    p.set_defaults(func=_cmd_path2ldr)

    p = sub.add_parser("sample", help="sample collision-free build sequences from models")
    p.add_argument("input", help=".ldr file, directory of .ldr files, or .npz graph batch")
    p.add_argument("-o", "--output", help="output jsonl (default stdout)")
    p.add_argument("--n", type=int, default=1, help="samples per connected component")
    p.add_argument("--seed", type=int, default=None, help="deterministic sampling (default: random)")
    p.set_defaults(func=_cmd_sample)

    p = sub.add_parser("score", help="score generated samples for parsability and collisions")
    p.add_argument("rest", nargs=argparse.REMAINDER, help="arguments for the scorer")
    p.set_defaults(func=_cmd_score)

    p = sub.add_parser("fetch-meshes", help="download the collision meshes into the data dir")
    p.add_argument("--dest", help="extract here instead of <data dir>/inset")
    p.set_defaults(func=_cmd_fetch_meshes)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
