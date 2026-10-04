"""Collision checking over convex part colliders.

Each part is a compound of convex hulls. Bullet computes penetration depths; overlaps up to
TAU (1 LDU by default) are allowed after subtracting Bullet's collision margins.

Colliders are read from <data dir>/cvx/, where the data dir is $BRICKNET_DATA if set, else the
platformdirs user data dir; `python -m bricknet fetch-meshes` downloads them there. Parts without
colliders remain in the scene and warn on each addition, but cannot be collision-checked.
"""

import os
import sys
import threading
from functools import lru_cache
from math import floor
from pathlib import Path

import numpy as np
import pybullet as p

from .data import load_catalog

TAU = 1.0  # Total pairwise overlap allowance
_MARGIN_BIAS = 0.002  # Bullet convex margin, 0.001 per shape
_CELL = 40.0  # spatial-hash cell, LDU
_CLIENT = None
_LOCK = threading.RLock()  # one global pybullet world


def data_dir() -> Path:
    """Root for the large external data (collision meshes)."""
    env = os.environ.get("BRICKNET_DATA")
    if env:
        return Path(env)
    import platformdirs

    return Path(platformdirs.user_data_dir("bricknet"))


def _client() -> int:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = p.connect(p.DIRECT)
    return _CLIENT


@lru_cache(maxsize=None)
def _collider_files(root: Path) -> frozenset[str]:
    """Installed collider filenames, cached per data directory."""
    try:
        with os.scandir(root) as entries:
            names = frozenset(entry.name for entry in entries if entry.name.endswith(".cvx.obj") and entry.is_file())
    except (FileNotFoundError, NotADirectoryError):
        names = frozenset()
    if not names:
        raise RuntimeError(
            f"Collider library is not installed at {root}. "
            "Run `python -m bricknet fetch-meshes` or set BRICKNET_DATA to an installed data directory.",
        )
    return names


def _path(part_id: int) -> Path:
    path = data_dir() / "cvx" / f"{load_catalog().id_to_stem[part_id]}.cvx.obj"
    if path.name not in _collider_files(path.parent):
        raise FileNotFoundError(f"Missing collider: {path}")
    return path


@lru_cache(maxsize=None)
def _bounds(part_id: int):
    """(local_lo, local_hi) from the OBJ header, expanded by Bullet's collision margin."""
    with _path(part_id).open("rb") as f:
        v = np.array(f.readline().split()[2:], dtype=np.float64)
    return tuple(v[:3] - _MARGIN_BIAS), tuple(v[3:] + _MARGIN_BIAS)


@lru_cache(maxsize=None)
def _shape(part_id: int) -> int:
    """Compound collision shape, one convex hull per OBJ `o` group."""
    return p.createCollisionShape(p.GEOM_MESH, fileName=str(_path(part_id)), physicsClientId=_client())


@lru_cache(maxsize=None)
def _probe(part_id: int, slot: int) -> int:
    """Reusable query body sharing the part's cached collision shape."""
    return p.createMultiBody(0, _shape(part_id), physicsClientId=_client())


def _pose(mat: np.ndarray):
    """(position, xyzw quaternion) from a 4x4 placement matrix."""
    m = mat[:3, :3]
    xx, yy, zz = m[0, 0], m[1, 1], m[2, 2]
    trace = xx + yy + zz
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        q = (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s
    elif xx > yy and xx > zz:
        s = np.sqrt(1.0 + xx - yy - zz) * 2
        q = 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s
    elif yy > zz:
        s = np.sqrt(1.0 + yy - xx - zz) * 2
        q = (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s
    else:
        s = np.sqrt(1.0 + zz - xx - yy) * 2
        q = (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s
    return mat[:3, 3].tolist(), q


def _place(body: int, pose) -> None:
    p.resetBasePositionAndOrientation(body, pose[0], pose[1], physicsClientId=_client())


def _depth(ba: int, bb: int, distance: float = 0.0) -> float:
    """Deepest overlap in LDU, 0 when clear. Negative `distance` prunes shallow contacts for
    threshold-only queries; Bullet's collision margins are subtracted from the depth."""
    # PyBullet can segfault when a single query returns more contacts than its native buffer holds.
    cps = p.getClosestPoints(ba, bb, distance=distance, physicsClientId=_client())
    return max(-min(c[8] for c in cps) - _MARGIN_BIAS, 0.0) if cps else 0.0


def _world_aabb(lo, hi, mat):
    rot, t = mat[:3, :3], mat[:3, 3]
    a, b = rot * lo, rot * hi
    return tuple(t + np.minimum(a, b).sum(1)), tuple(t + np.maximum(a, b).sum(1))


def penetration(id_a, mat_a, id_b, mat_b) -> float:
    """Deepest interpenetration between two placed parts, in LDU.
    Raises FileNotFoundError for a missing part collider."""
    mat_a, mat_b = np.asarray(mat_a, np.float64), np.asarray(mat_b, np.float64)
    with _LOCK:
        ba, bb = _probe(id_a, 0), _probe(id_b, 1)
        _place(ba, _pose(mat_a))
        _place(bb, _pose(mat_b))
        return _depth(ba, bb)


class CollisionScene:
    """Incremental collision world: parts keyed by part_id, poses are 4x4 LDU world matrices.
    Parts without colliders remain placed, warn on each addition, and are not collision-checked."""

    def __init__(self, tau: float | None = None):
        self.tau = TAU if tau is None else tau
        self._parts = []
        self._poses = []  # (position, quaternion): a placement's pose never changes, convert once
        self._lo = []
        self._hi = []
        self._grid = {}
        self._unchecked = False

    def _candidates(self, lo, hi):
        inv = 1.0 / _CELL
        seen = set()
        for x in range(floor(lo[0] * inv), floor(hi[0] * inv) + 1):
            for y in range(floor(lo[1] * inv), floor(hi[1] * inv) + 1):
                for z in range(floor(lo[2] * inv), floor(hi[2] * inv) + 1):
                    seen.update(self._grid.get((x, y, z), ()))
        for j in sorted(seen):
            jlo, jhi = self._lo[j], self._hi[j]
            if (
                lo[0] <= jhi[0]
                and hi[0] >= jlo[0]
                and lo[1] <= jhi[1]
                and hi[1] >= jlo[1]
                and lo[2] <= jhi[2]
                and hi[2] >= jlo[2]
            ):
                yield j

    def check(self, part_id: int, mat: np.ndarray, *, exclude: int = -1, first_only: bool = False) -> list[int]:
        """Checked scene indices colliding at this pose (the part is not added).
        Pairs involving an unavailable collider are not checked."""
        try:
            llo, lhi = _bounds(part_id)
        except FileNotFoundError:
            return []
        mat = np.asarray(mat, dtype=np.float64)
        with _LOCK:
            probe = _probe(part_id, 0)
            _place(probe, _pose(mat))
            hits = []
            for j in self._candidates(*_world_aabb(llo, lhi, mat)):
                if j == exclude:
                    continue
                other = _probe(self._parts[j], 1)  # slot 1: distinct body when both are one part
                _place(other, self._poses[j])
                if _depth(probe, other, -(self.tau + _MARGIN_BIAS)) > self.tau:
                    hits.append(j)
                    if first_only:
                        break
            return hits

    def add(self, part_id: int, mat: np.ndarray) -> int:
        """Place the part; returns its scene index. Warn on each addition missing a collider."""
        mat = np.asarray(mat, dtype=np.float64)
        try:
            lo, hi = _world_aabb(*_bounds(part_id), mat)
        except FileNotFoundError as e:
            print(f"warning: {e} Part added without collision checking.", file=sys.stderr)
            lo = hi = None
            self._unchecked = True
        idx = len(self._parts)
        self._parts.append(part_id)
        self._poses.append(_pose(mat))
        self._lo.append(lo)
        self._hi.append(hi)
        if lo is None:
            return idx
        inv = 1.0 / _CELL
        for x in range(floor(lo[0] * inv), floor(hi[0] * inv) + 1):
            for y in range(floor(lo[1] * inv), floor(hi[1] * inv) + 1):
                for z in range(floor(lo[2] * inv), floor(hi[2] * inv) + 1):
                    self._grid.setdefault((x, y, z), []).append(idx)
        return idx

    def close(self) -> None:
        """Clear scene placements and spatial index, retaining shared Bullet bodies."""
        self._parts.clear()
        self._poses.clear()
        self._lo.clear()
        self._hi.clear()
        self._grid.clear()
        self._unchecked = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def colliding_pairs(part_ids, mats) -> list[tuple[int, int]] | None:
    """Colliding index pairs (i < j) over absolute placements, no exclusions.
    Returns None if any placement lacks a collider."""
    with CollisionScene() as scene:
        pairs = []
        for i, (pid, mat) in enumerate(zip(part_ids, mats)):
            mat = np.asarray(mat, dtype=np.float64)
            pairs.extend((j, i) for j in scene.check(pid, mat))
            scene.add(pid, mat)
        return None if scene._unchecked else pairs


def check_placements(part_ids, mats, fixed_parents: dict | None = None) -> list[int] | None:
    """Indices whose part collides under autoregressive placement (coincident duplicates count;
    children in fixed_parents ignore that parent, which they intentionally overlap). Colliding
    parts stay placed: scoring, not rejection. Returns None if any placement lacks a collider."""
    fixed_parents = fixed_parents or {}
    with CollisionScene() as scene:
        seen, bad = set(), []
        for i, (pid, mat) in enumerate(zip(part_ids, mats)):
            mat = np.asarray(mat, dtype=np.float64)
            key = (pid, mat.tobytes())
            hit = key in seen or bool(scene.check(pid, mat, exclude=fixed_parents.get(i, -1), first_only=True))
            seen.add(key)
            scene.add(pid, mat)
            if hit:
                bad.append(i)
        return None if scene._unchecked else bad


def first_collision(part_ids, mats, fixed_parents: dict | None = None) -> int | None:
    """Index of the first colliding placement (len(part_ids) when clean); nothing past it is placed.
    Returns None if an examined placement lacks a collider."""
    fixed_parents = fixed_parents or {}
    with CollisionScene() as scene:
        seen = set()
        for i, (pid, mat) in enumerate(zip(part_ids, mats)):
            mat = np.asarray(mat, dtype=np.float64)
            key = (pid, mat.tobytes())
            if key in seen or scene.check(pid, mat, exclude=fixed_parents.get(i, -1), first_only=True):
                return None if scene._unchecked else i
            seen.add(key)
            scene.add(pid, mat)
        return None if scene._unchecked else len(part_ids)
