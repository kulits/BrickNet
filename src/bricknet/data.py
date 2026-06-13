"""Catalog, connector, and alias loaders over the bundled data files."""

import json
import lzma
import os
from collections import Counter
from functools import lru_cache
from importlib.resources import files
from types import MappingProxyType

import numpy as np

from .core import Catalog

# catalog version: directory under _data/ holding the part vocabulary and connector labels
_DATA = files("bricknet") / "_data" / os.environ.get("BRICKNET_CATALOG", "v1")


@lru_cache(maxsize=1)
def _part_names() -> dict:
    return json.loads((_DATA / "part_names.json").read_text())  # {stem: name}, part_id == file-order index


@lru_cache(maxsize=1)
def _load_all_connectors():
    labels = json.loads(lzma.decompress((_DATA / "labels.json.xz").read_bytes()))
    ranges = []  # (part_id, kind, sub, pol, start, count)
    raw_rows = []  # (x, y, z, pitch_deg, roll_deg, yaw_deg)
    axle_lengths = []  # parallel to raw_rows; non-zero only for axle connectors
    for part_id, stem in enumerate(_part_names()):
        kinds = labels[stem + ".dat"]
        for kind in sorted(kinds):
            for sub in sorted(kinds[kind]):
                val = kinds[kind][sub]
                for pol, rows in [(p, val[p]) for p in sorted(val)] if isinstance(val, dict) else [(None, val)]:
                    start = len(raw_rows)
                    for row in rows:
                        raw_rows.append(
                            (
                                row[0],
                                row[1],
                                row[2],
                                row[3] if len(row) > 3 else 0,
                                row[4] if len(row) > 4 else 0,
                                row[5] if kind == "fixed" else 0,  # col 5: yaw, fixed connectors only
                            )
                        )
                        axle_lengths.append(row[5] if kind == "axle" else 0)
                    ranges.append((part_id, kind, sub, pol, start, len(raw_rows) - start))

    raw_arr = np.array(raw_rows)
    angles = np.radians(raw_arr[:, 3:])
    c, s = np.cos(angles), np.sin(angles)
    cx, cz, cy, sx, sz, sy = c[:, 0], c[:, 1], c[:, 2], s[:, 0], s[:, 1], s[:, 2]
    px, py, pz = raw_arr[:, 0], raw_arr[:, 1], raw_arr[:, 2]
    z = np.zeros_like(cx)
    frames = np.array(
        [
            [cy * cz + sy * sx * sz, -cy * sz + sy * sx * cz, sy * cx, px],
            [cx * sz, cx * cz, -sx, py],
            [-sy * cz + cy * sx * sz, sy * sz + cy * sx * cz, cy * cx, pz],
            [z, z, z, z + 1],
        ]
    )
    frames.flags.writeable = False  # freeze the base, not a view: the cached frames are shared
    frames = frames.transpose(2, 0, 1)

    frames_dict: dict[int, dict[tuple[str, str | None], list[np.ndarray]]] = {}
    conn_list: dict[int, list] = {}
    counts: dict[int, dict[tuple, int]] = {}
    for part_id, kind, sub, pol, start, count in ranges:
        frames_dict.setdefault(part_id, {}).setdefault((sub, pol), []).extend(frames[start : start + count])
        cl = conn_list.setdefault(part_id, [])
        sc = counts.setdefault(part_id, {})
        for k in range(count):
            idx = sc.get((sub, pol), 0)
            sc[(sub, pol)] = idx + 1
            cl.append((kind, sub, pol, idx, axle_lengths[start + k]))
    frozen_frames = MappingProxyType(
        {pid: MappingProxyType({key: tuple(v) for key, v in groups.items()}) for pid, groups in frames_dict.items()}
    )
    return frozen_frames, MappingProxyType({p: tuple(v) for p, v in conn_list.items()})


@lru_cache(maxsize=1)
def load_aliases() -> dict[str, tuple[str, np.ndarray]]:
    """{src stem: (canonical stem, 4x4 wrapper transform)}; substitute via T_new = T_part @ T."""
    out = {}
    for r in json.loads(lzma.decompress((_DATA / "part_aliases.json.xz").read_bytes()))["rows"]:
        v = r["final_matrix_3x4"]  # LDraw line-type-1 order: x y z a b c d e f g h i
        m = np.eye(4)
        m[:3, 3] = v[0:3]
        m[:3, :3] = np.array(v[3:12]).reshape(3, 3)
        m.flags.writeable = False
        out[r["src"]] = (r["dst"], m)
    return MappingProxyType(out)


def load_connectors() -> dict[int, dict[tuple[str, str | None], tuple[np.ndarray, ...]]]:
    """{part_id: {(sub, pol): (frame_4x4, ...)}} per subgroup."""
    return _load_all_connectors()[0]


def load_connector_list() -> dict[int, tuple]:
    """{part_id: ((kind, sub, pol, subgroup_idx, length), ...)} in canonical flat order."""
    return _load_all_connectors()[1]


@lru_cache(maxsize=1)
def load_catalog() -> Catalog:
    part_names = _part_names()
    id_to_stem = tuple(part_names)
    id_to_name = tuple(part_names.values())
    stem_to_id = {s: i for i, s in enumerate(id_to_stem)}
    name_to_id = {n: i for i, n in enumerate(id_to_name)}
    color_to_code = json.loads((_DATA / "color_names.json").read_text())
    code_to_color = {v: k for k, v in color_to_code.items()}
    conn_counts = {
        pid: MappingProxyType(dict(Counter((c[1], c[2]) for c in conns)))
        for pid, conns in load_connector_list().items()
    }
    return Catalog(
        MappingProxyType(conn_counts),
        id_to_stem,
        MappingProxyType(stem_to_id),
        id_to_name,
        MappingProxyType(name_to_id),
        MappingProxyType(color_to_code),
        MappingProxyType(code_to_color),
    )
