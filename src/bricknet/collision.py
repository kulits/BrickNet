"""Collision checking over the inset part meshes.

Meshes are pre-inset inward 0.25 LDU per surface, so mating parts sit ~0.5 LDU apart and
"collision" means real interpenetration; the runtime test is exact triangle intersection
with no margin parameter.

Meshes are read from <data dir>/inset/, where the data dir is $BRICKNET_DATA if set, else the
platformdirs user data dir; `python -m bricknet fetch-meshes` downloads them there.
"""

import os
from functools import lru_cache
from math import floor
from pathlib import Path

import meshlib.mrmeshpy as mm
import numpy as np

from .data import load_catalog


def data_dir() -> Path:
    """Root for the large external data (collision meshes)."""
    env = os.environ.get("BRICKNET_DATA")
    if env:
        return Path(env)
    import platformdirs

    return Path(platformdirs.user_data_dir("bricknet"))


_CELL = 40.0  # spatial-hash cell, LDU
# findCollidingTriangles reports no hits for exactly coplanar meshes (it detects triangle
# crossings), so every kernel call perturbs the relative translation by a hash-deterministic
# unit vector * eps: below any LEGO feature, above float noise, deterministic per transform.
_PERTURB_EPS = 1e-6

_Z_FLIP = mm.AffineXf3f()
_Z_FLIP.A = mm.Matrix3f(mm.Vector3f(1, 0, 0), mm.Vector3f(0, 1, 0), mm.Vector3f(0, 0, -1))


@lru_cache(maxsize=None)
def _mesh(part_id):
    """(mesh, local_lo, local_hi). PLYs store Z negated, flipped here."""
    path = data_dir() / "inset" / f"{load_catalog().id_to_stem[part_id]}.ply"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} -- run `python -m bricknet fetch-meshes` or set BRICKNET_DATA"
            " (parse-only scoring works without meshes: `score --no-collision`)"
        )
    mesh = mm.loadMesh(str(path))
    mesh.transform(_Z_FLIP)
    box = mesh.computeBoundingBox()
    return mesh, (box.min.x, box.min.y, box.min.z), (box.max.x, box.max.y, box.max.z)


def _xf(mat) -> mm.AffineXf3f:
    f = mat[:3, :4].ravel().tolist()
    xf = mm.AffineXf3f()
    xf.A = mm.Matrix3f(mm.Vector3f(f[0], f[1], f[2]), mm.Vector3f(f[4], f[5], f[6]), mm.Vector3f(f[8], f[9], f[10]))
    xf.b = mm.Vector3f(f[3], f[7], f[11])
    return xf


def _perturbed(b2a: mm.AffineXf3f) -> mm.AffineXf3f:
    buf = np.empty(12)
    a, b = b2a.A, b2a.b
    buf[0:3] = a.x.x, a.x.y, a.x.z
    buf[3:6] = a.y.x, a.y.y, a.y.z
    buf[6:9] = a.z.x, a.z.y, a.z.z
    buf[9:12] = b.x, b.y, b.z
    rng = np.random.default_rng(int.from_bytes(buf.tobytes()[:8], "little") & 0xFFFFFFFF)
    d = rng.standard_normal(3)
    d /= np.linalg.norm(d)
    out = mm.AffineXf3f()
    out.A = mm.Matrix3f(a.x, a.y, a.z)
    out.b = mm.Vector3f(b.x + d[0] * _PERTURB_EPS, b.y + d[1] * _PERTURB_EPS, b.z + d[2] * _PERTURB_EPS)
    return out


def _world_aabb(lo, hi, mat):
    rot, t = mat[:3, :3], mat[:3, 3]
    corners = rot * np.array([lo, hi])[:, None, :]  # (2, 3, 3): per output axis, min/max contributions
    return tuple(t + corners.min(axis=0).sum(axis=1)), tuple(t + corners.max(axis=0).sum(axis=1))


def _obb(lo, hi, mat):
    """(center(3), half_extents(3), unit axes(3x3 columns)) in world space."""
    lo, hi = np.asarray(lo), np.asarray(hi)
    rot, t = mat[:3, :3], mat[:3, 3]
    center = rot @ ((lo + hi) * 0.5) + t
    scale = np.linalg.norm(rot, axis=0)
    axes = rot / np.where(scale > 0, scale, 1.0)
    return center, (hi - lo) * 0.5 * scale, axes


def _obb_separated(a, b) -> bool:
    """15-axis separating-axis test; True when the OBBs do not overlap."""
    ca, ea, ra = a
    cb, eb, rb = b
    c = ra.T @ rb
    t = ra.T @ (cb - ca)
    ac = np.abs(c) + 1e-6  # epsilon guards near-parallel edge pairs
    if (np.abs(t) > ea + ac @ eb).any():
        return True
    if (np.abs(t @ c) > ea @ ac + eb).any():
        return True
    for i in range(3):
        i1, i2 = (i + 1) % 3, (i + 2) % 3
        for j in range(3):
            j1, j2 = (j + 1) % 3, (j + 2) % 3
            lhs = abs(t[i2] * c[i1, j] - t[i1] * c[i2, j])
            rhs = ea[i1] * ac[i2, j] + ea[i2] * ac[i1, j] + eb[j1] * ac[i, j2] + eb[j2] * ac[i, j1]
            if lhs > rhs:
                return True
    return False


class CollisionScene:
    """Incremental collision world: parts keyed by part_id, poses are 4x4 LDU world matrices."""

    def __init__(self):
        self._meshes = []
        self._xf_invs = []
        self._lo = []
        self._hi = []
        self._obbs = []
        self._grid = {}

    def _candidates(self, lo, hi):
        inv = 1.0 / _CELL
        seen = set()
        for x in range(floor(lo[0] * inv), floor(hi[0] * inv) + 1):
            for y in range(floor(lo[1] * inv), floor(hi[1] * inv) + 1):
                for z in range(floor(lo[2] * inv), floor(hi[2] * inv) + 1):
                    seen.update(self._grid.get((x, y, z), ()))
        for j in seen:
            jlo, jhi = self._lo[j], self._hi[j]
            if all(lo[i] <= jhi[i] and hi[i] >= jlo[i] for i in range(3)):
                yield j

    def check(self, part_id: int, mat: np.ndarray, *, exclude: int = -1, first_only: bool = False) -> list[int]:
        """Scene indices colliding with the part at this pose (the part is not added)."""
        mesh, llo, lhi = _mesh(part_id)
        mat = np.asarray(mat, dtype=np.float64)
        xf = _xf(mat)
        obb = _obb(llo, lhi, mat)
        hits = []
        for j in self._candidates(*_world_aabb(llo, lhi, mat)):
            if j == exclude or _obb_separated(obb, self._obbs[j]):
                continue
            b2a = self._xf_invs[j] * xf
            if len(mm.findCollidingTriangles(self._meshes[j], mesh, _perturbed(b2a), True)) > 0:
                hits.append(j)
                if first_only:
                    break
        return hits

    def add(self, part_id: int, mat: np.ndarray) -> int:
        """Place the part; returns its scene index."""
        mesh, llo, lhi = _mesh(part_id)
        mat = np.asarray(mat, dtype=np.float64)
        lo, hi = _world_aabb(llo, lhi, mat)
        idx = len(self._meshes)
        self._meshes.append(mesh)
        self._xf_invs.append(_xf(mat).inverse())
        self._lo.append(lo)
        self._hi.append(hi)
        self._obbs.append(_obb(llo, lhi, mat))
        inv = 1.0 / _CELL
        for x in range(floor(lo[0] * inv), floor(hi[0] * inv) + 1):
            for y in range(floor(lo[1] * inv), floor(hi[1] * inv) + 1):
                for z in range(floor(lo[2] * inv), floor(hi[2] * inv) + 1):
                    self._grid.setdefault((x, y, z), []).append(idx)
        return idx


def colliding_pairs(part_ids, mats) -> list[tuple[int, int]]:
    """Colliding index pairs (i < j) over absolute placements, no exclusions."""
    scene = CollisionScene()
    pairs = []
    for i, (pid, mat) in enumerate(zip(part_ids, mats)):
        mat = np.asarray(mat, dtype=np.float64)
        pairs.extend((j, i) for j in scene.check(pid, mat))
        scene.add(pid, mat)
    return pairs


def check_placements(part_ids, mats, fixed_parents: dict | None = None) -> list[int]:
    """Indices whose part collides under autoregressive placement (coincident duplicates count;
    children in fixed_parents ignore that parent, which they intentionally overlap). Colliding
    parts stay placed: scoring, not rejection."""
    fixed_parents = fixed_parents or {}
    scene = CollisionScene()
    seen, bad = set(), []
    for i, (pid, mat) in enumerate(zip(part_ids, mats)):
        mat = np.asarray(mat, dtype=np.float64)
        key = (pid, mat.tobytes())
        hit = key in seen or bool(scene.check(pid, mat, exclude=fixed_parents.get(i, -1), first_only=True))
        seen.add(key)
        scene.add(pid, mat)
        if hit:
            bad.append(i)
    return bad


def first_collision(part_ids, mats, fixed_parents: dict | None = None) -> int:
    """Index of the first colliding placement (len(part_ids) when clean); nothing past it is placed."""
    fixed_parents = fixed_parents or {}
    scene = CollisionScene()
    seen = set()
    for i, (pid, mat) in enumerate(zip(part_ids, mats)):
        mat = np.asarray(mat, dtype=np.float64)
        key = (pid, mat.tobytes())
        if key in seen or scene.check(pid, mat, exclude=fixed_parents.get(i, -1), first_only=True):
            return i
        seen.add(key)
        scene.add(pid, mat)
    return len(part_ids)
