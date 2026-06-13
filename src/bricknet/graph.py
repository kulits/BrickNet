"""LDR parsing, connector pairing, tree sampling, realization, and graph .npz I/O."""

import math
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .data import load_aliases, load_catalog, load_connector_list, load_connectors
from .core import (
    AxleEdge,
    AxleSub,
    BallEdge,
    FixedEdge,
    Graph,
    HingeEdge,
    Part,
    Tree,
    StudEdge,
    StudSub,
    _AXLE_MATES,
    _STUD_MATES,
)
from .tree import RX_PI, ry4

_POS_THR = 3.0
_AXIS_THR = 0.95
_AXLE_AXIS_THR = 0.9
_AXLE_PERP_THR = 2.05
_AXLE_DEFAULT_RAD = 4.0
_AXLE_MIN_OVERLAP = 4.0
_AXLE_QUERY_PAD = 0.6
_FIXED_AXIS_THR = 0.9999
_FIXED_FRAME_TRACE = 2.99
_FIXED_ORIENTED = frozenset({"0", "1", "2", "3"})  # fixed subtypes requiring strict orientation; others rotate freely
_AXLE_A_SIDE = frozenset({"socket", "cross", "clip"})


class Kind(IntEnum):
    stud = 0
    hole = 1
    hinge = 2
    fixed = 3
    ball = 4
    axle = 5


# stud and hole both map to family "stud"
_FAM_NAMES = ["stud" if k in (Kind.stud, Kind.hole) else k.name for k in Kind]
# AxleSub codes shifted by len(StudSub) into a shared namespace
_SUB_ENC = {e.name: e.value for e in StudSub} | {e.name: e.value + len(StudSub) for e in AxleSub}
_POL_ENC = {"in": 0, "on": 1}
_N_SUBS = len(_SUB_ENC)


def _sub_enc(s):
    """Digit subtypes encode as 100 + int."""
    return _SUB_ENC[s] if s in _SUB_ENC else 100 + int(s)


def _build_mate_mat(pairs):
    mat = np.zeros((_N_SUBS, _N_SUBS), dtype=bool)
    for a, b in pairs:
        mat[_SUB_ENC[a], _SUB_ENC[b]] = mat[_SUB_ENC[b], _SUB_ENC[a]] = True
    return mat


_STUD_MATE_MAT = _build_mate_mat(_STUD_MATES)
_AXLE_MATE_MAT = _build_mate_mat(_AXLE_MATES)

_EDGE_DT = np.dtype(
    [
        ("a", "i4"),
        ("b", "i4"),
        ("a_conn", "i4"),
        ("b_conn", "i4"),
        ("family", "U8"),
        ("yaw", "f8"),
        ("flip", "?"),
        ("rot", "f8", 3),
    ]
)


@dataclass(frozen=True)
class PartInfo:
    conns: tuple  # (kind, sub, pol, subgroup_idx, length) per connector
    frames: np.ndarray  # (N, 4, 4) part-local
    meta: np.ndarray  # (N, 4) of (kind, sub_enc, pol_enc, length)
    flat_of: dict  # (sub, pol, subgroup_idx) -> flat connector index


@dataclass(frozen=True)
class ConnArrays:
    """Per-connector PartInfo columns flattened over a model's parts."""

    part_of: np.ndarray
    kind: np.ndarray
    sub: np.ndarray
    pol: np.ndarray
    length: np.ndarray
    conn_idx: np.ndarray  # within-part flat index
    frames: np.ndarray  # part-local


@lru_cache(maxsize=None)
def _part_info(part_id):
    """Connectors in canonical flat order."""
    conns = load_connector_list()[part_id]
    conn_frames = load_connectors()[part_id]
    frames = np.array([conn_frames[(c[1], c[2])][c[3]] for c in conns])
    meta = np.array([(Kind[c[0]], _sub_enc(c[1]), _POL_ENC.get(c[2], -1), c[4]) for c in conns])
    flat_of = {(c[1], c[2], c[3]): k for k, c in enumerate(conns)}
    return PartInfo(conns, frames, meta, flat_of)


def _empty_graph(part_ids, colors, mats):
    """Every part is its own component."""
    return Graph(part_ids, colors, mats, np.empty(0, dtype=_EDGE_DT), tuple((i,) for i in range(len(part_ids))))


def _find_pairs(part_ids, mats, ca):
    """Returns (raw_pairs, wf, axes): raw_pairs (M, 4) of (part_i, conn_i, part_j, conn_j),
    wf world frames, axes negated normalized connector Y-axes."""
    part_of = ca.part_of
    kind_arr, sub_arr, pol_arr, f_len = ca.kind, ca.sub, ca.pol, ca.length
    f_conn, flat_frames = ca.conn_idx, ca.frames

    wf = mats[part_of] @ flat_frames
    wf_pos = wf[:, :3, 3]
    col1 = wf[:, :3, 1]
    norm_sq = np.einsum("ij,ij->i", col1, col1)[:, None]
    norm_sq[norm_sq == 0] = 1.0
    axes = -col1 / np.sqrt(norm_sq)

    pairs = cKDTree(wf_pos).query_pairs(r=_POS_THR, output_type="ndarray")

    # axles mate over larger radii than _POS_THR, so query separately
    axle_indices = np.where(kind_arr == Kind.axle)[0]
    if len(axle_indices) > 1:
        apos = wf_pos[axle_indices]
        arad = np.maximum(f_len[axle_indices], _AXLE_DEFAULT_RAD)
        atree = cKDTree(apos)
        max_rad = float(arad.max())
        extra = set()
        for li in range(len(axle_indices)):
            for lj in atree.query_ball_point(apos[li], r=arad[li] + max_rad + _AXLE_PERP_THR + _AXLE_QUERY_PAD):
                if lj > li:
                    extra.add((int(axle_indices[li]), int(axle_indices[lj])))
        if extra:
            axle_pairs = np.array(sorted(extra), dtype=np.intp)
            pairs = np.vstack([pairs, axle_pairs]) if len(pairs) > 0 else axle_pairs
            # a close axle pair is returned by both query_pairs and this search; keep first occurrence
            pairs = pairs[np.sort(np.unique(pairs, axis=0, return_index=True)[1])]

    if len(pairs) == 0:
        return np.empty((0, 4), dtype=np.intp), wf, axes

    pi, pj = pairs[:, 0], pairs[:, 1]
    ki, kj = kind_arr[pi], kind_arr[pj]
    si, sj = sub_arr[pi], sub_arr[pj]
    poli, polj = pol_arr[pi], pol_arr[pj]
    diff_part = part_of[pi] != part_of[pj]

    is_stud_hole = diff_part & (((ki == Kind.stud) & (kj == Kind.hole)) | ((ki == Kind.hole) & (kj == Kind.stud)))
    # digit subs (100+) alias mod _N_SUBS; harmless: the stud/axle mate masks admit only named subs
    sub_i, sub_j = si.astype(np.intp) % _N_SUBS, sj.astype(np.intp) % _N_SUBS
    stud_ok = is_stud_hole & _STUD_MATE_MAT[sub_i, sub_j]

    # opposite polarity: one "in", one "on"
    is_same_kind = diff_part & (ki == kj) & (ki >= Kind.hinge)
    pol_match = is_same_kind & (si == sj) & (poli >= 0) & (polj >= 0) & (poli != polj)
    hinge_ball_ok = pol_match & (ki != Kind.axle) & (ki != Kind.fixed)

    # stud/hole: signed dot (parallel only); hinge: abs dot (allows anti-parallel)
    hinge_needs_axis = hinge_ball_ok & (ki < Kind.ball)
    if stud_ok.any() or hinge_needs_axis.any():
        raw_dots = np.einsum("ij,ij->i", axes[pi], axes[pj])
        stud_ok &= raw_dots >= _AXIS_THR
        hinge_ball_ok &= ~hinge_needs_axis | (np.abs(raw_dots) >= _AXIS_THR)

    result_parts = []
    fast_mask = stud_ok | hinge_ball_ok
    if fast_mask.any():
        matched_i, matched_j = pi[fast_mask], pj[fast_mask]
        result_parts.append(
            np.column_stack([part_of[matched_i], f_conn[matched_i], part_of[matched_j], f_conn[matched_j]])
        )

    fixed_ok = pol_match & (ki == Kind.fixed)
    axle_ok = is_same_kind & (ki == Kind.axle) & _AXLE_MATE_MAT[sub_i, sub_j]
    if (fixed_ok | axle_ok).any():
        po_list = part_of.tolist()
        fc_list = f_conn.tolist()
        pi_list, pj_list = pi.tolist(), pj.tolist()
        extra_pairs = []
        for idx in np.where(fixed_ok | axle_ok)[0]:
            ci, cj = pi_list[idx], pj_list[idx]
            conn_i = _part_info(part_ids[po_list[ci]]).conns[fc_list[ci]]
            conn_j = _part_info(part_ids[po_list[cj]]).conns[fc_list[cj]]
            if fixed_ok[idx]:
                dp = wf_pos[ci] - wf_pos[cj]
                if float(dp @ dp) > _POS_THR**2:
                    continue
                if conn_i[1] in _FIXED_ORIENTED and conn_j[1] in _FIXED_ORIENTED:
                    if (
                        axes[ci] @ axes[cj] < _FIXED_AXIS_THR
                        or np.trace(wf[ci, :3, :3].T @ wf[cj, :3, :3]) < _FIXED_FRAME_TRACE
                    ):
                        continue
            else:
                axis_dot = abs(float(axes[ci] @ axes[cj]))
                if axis_dot < _AXLE_AXIS_THR:
                    continue
                ref = axes[ci] if conn_i[1] not in _AXLE_A_SIDE or conn_j[1] in _AXLE_A_SIDE else axes[cj]
                delta = wf_pos[cj] - wf_pos[ci]
                proj = float(delta @ ref)
                perp = delta - proj * ref
                perp_dist_sq = float(perp @ perp)
                if perp_dist_sq > _POS_THR**2:
                    continue
                overlap = max(f_len[ci], _AXLE_DEFAULT_RAD) + max(f_len[cj], _AXLE_DEFAULT_RAD) - abs(proj)
                if (
                    overlap < _AXLE_MIN_OVERLAP
                    and max(0.0, _POS_THR**2 - perp_dist_sq) ** 0.5 < _AXLE_MIN_OVERLAP - overlap - 1e-4
                ):
                    continue
            extra_pairs.append([po_list[ci], fc_list[ci], po_list[cj], fc_list[cj]])
        if extra_pairs:
            result_parts.append(np.array(extra_pairs, dtype=np.intp))

    raw_pairs = np.vstack(result_parts) if result_parts else np.empty((0, 4), dtype=np.intp)
    return raw_pairs, wf, axes


def parse_ldr(text: str) -> Graph:
    type1 = [line.split() for line in text.split("\n") if line.startswith("1 ")]
    if not type1:
        return _empty_graph((), (), np.empty((0, 4, 4)))

    n = len(type1)
    stems = [t[14].lower().removesuffix(".dat") for t in type1]
    colors = tuple(int(t[1]) for t in type1)
    raw = np.array([t[2:14] for t in type1], dtype=np.float64)
    mats = np.zeros((n, 4, 4))
    mats[:, :3, :3] = raw[:, 3:].reshape(n, 3, 3)
    mats[:, :3, 3] = raw[:, :3]
    mats[:, 3, 3] = 1.0
    aliases = load_aliases()
    for i, s in enumerate(stems):
        if s in aliases:
            stems[i], wrap = aliases[s]
            mats[i] = mats[i] @ wrap
    U, _, Vt = np.linalg.svd(mats[:, :3, :3])
    mats[:, :3, :3] = U @ Vt
    part_ids = tuple(load_catalog().stem_to_id[s] for s in stems)

    infos = [_part_info(pid) for pid in part_ids]
    part_counts = np.array([len(info.meta) for info in infos])
    part_of = np.repeat(np.arange(n), part_counts)
    meta = np.concatenate([info.meta for info in infos])
    kind_arr, sub_arr, pol_arr, f_len = meta[:, 0], meta[:, 1], meta[:, 2], meta[:, 3]
    cumulative = np.cumsum(part_counts)
    f_conn = np.arange(cumulative[-1], dtype=np.int64) - np.repeat(cumulative - part_counts, part_counts)
    part_offsets = np.concatenate([[0], cumulative])
    flat_frames = np.concatenate([info.frames for info in infos])

    # Canonical direction: stud=A, hole=B; polarity "in"=A; axle socket/cross/clip=A
    is_stud = kind_arr == Kind.stud
    is_pol_in = (pol_arr == _POL_ENC["in"]) & (kind_arr >= Kind.hinge)
    is_axle_a = (kind_arr == Kind.axle) & (
        (sub_arr == _SUB_ENC["socket"]) | (sub_arr == _SUB_ENC["cross"]) | (sub_arr == _SUB_ENC["clip"])
    )
    is_a = is_stud | is_pol_in | is_axle_a  # hole(1) can't satisfy any term, so no hole guard needed

    ca = ConnArrays(part_of, kind_arr, sub_arr, pol_arr, f_len, f_conn, flat_frames)
    raw_pairs, wf, axes = _find_pairs(part_ids, mats, ca)
    ne = len(raw_pairs)
    if ne == 0:
        return _empty_graph(part_ids, colors, mats)

    pi_r, ci_r, pj_r, cj_r = raw_pairs.T
    swap = ~is_a[(part_offsets[pi_r] + ci_r).astype(np.intp)]
    e_ap, e_ac = np.where(swap, pj_r, pi_r), np.where(swap, cj_r, ci_r)
    e_bp, e_bc = np.where(swap, pi_r, pj_r), np.where(swap, ci_r, cj_r)
    a_flat = (part_offsets[e_ap] + e_ac).astype(np.intp)
    b_flat = (part_offsets[e_bp] + e_bc).astype(np.intp)
    fam_idx = np.where(
        (kind_arr[a_flat] <= Kind.hole) & (kind_arr[b_flat] <= Kind.hole), Kind.stud, kind_arr[a_flat]
    ).astype(np.intp)

    components = _components_from_edges(n, e_ap, e_bp)

    wa, wb = wf[a_flat], wf[b_flat]
    yaws = np.arctan2(
        -np.einsum("ij,ij->i", wa[:, :3, 2], wb[:, :3, 0]),
        np.einsum("ij,ij->i", wa[:, :3, 0], wb[:, :3, 0]),
    )
    hinge_mask = fam_idx == Kind.hinge
    axle_mask = fam_idx == Kind.axle
    ball_mask = fam_idx == Kind.ball
    flips = np.zeros(ne, dtype=bool)
    axle_slides = np.zeros(ne)
    if hinge_mask.any():
        flips[hinge_mask] = np.einsum("ij,ij->i", wa[hinge_mask, :3, 1], wb[hinge_mask, :3, 1]) < 0
    if axle_mask.any():
        axle_axes = -wa[axle_mask, :3, 1]
        norm_sq = np.einsum("ij,ij->i", axle_axes, axle_axes)[:, None]
        norm_sq[norm_sq == 0] = 1.0
        axle_axes /= np.sqrt(norm_sq)
        flips[axle_mask] = np.einsum("ij,ij->i", axle_axes, -wb[axle_mask, :3, 1]) < 0
        axle_slides[axle_mask] = np.einsum("ij,ij->i", wb[axle_mask, :3, 3] - wa[axle_mask, :3, 3], axle_axes)
    # axle edges store the slide in the "yaw" field and the axle rotation in rot[0]
    yaw_vals = np.where(
        axle_mask, axle_slides, yaws
    )  # the axle loop below still reads yaws, so it keeps the raw rotations
    yaw_vals[fam_idx == Kind.fixed] = 0.0
    rots = [(0.0, 0.0, 0.0)] * ne
    for idx in np.where(axle_mask)[0]:
        rots[idx] = (float(yaws[idx]), 0.0, 0.0)
    if ball_mask.any():
        for idx in np.where(ball_mask)[0]:
            rots[idx] = tuple(map(float, Rotation.from_matrix(wa[idx, :3, :3].T @ wb[idx, :3, :3]).as_euler("xyz")))
            yaw_vals[idx] = 0.0
            flips[idx] = False

    edges = np.empty(ne, dtype=_EDGE_DT)
    edges["a"], edges["b"] = e_ap, e_bp
    edges["a_conn"], edges["b_conn"] = e_ac, e_bc
    edges["family"] = np.array(_FAM_NAMES)[fam_idx]
    edges["yaw"] = yaw_vals
    edges["flip"] = flips
    edges["rot"] = rots
    return Graph(part_ids, colors, mats, edges, components)


def _qdeg(rad):
    """Quantize radians to integer degrees in [0, 360)."""
    return round(np.degrees(rad)) % 360


def _make_tree_edge(ge, parent, child, forward, part_ids):
    family = str(ge["family"])
    yaw, flip, rot = float(ge["yaw"]), bool(ge["flip"]), ge["rot"]

    if forward:
        parent_idx, child_idx = (ge["a"], ge["a_conn"]), (ge["b"], ge["b_conn"])
    else:
        parent_idx, child_idx = (ge["b"], ge["b_conn"]), (ge["a"], ge["a_conn"])
    parent_conn = _part_info(part_ids[parent_idx[0]]).conns[parent_idx[1]]
    child_conn = _part_info(part_ids[child_idx[0]]).conns[child_idx[1]]

    if family in ("stud", "axle"):
        sub_cls = StudSub if family == "stud" else AxleSub
        parent_sub = sub_cls[parent_conn[1]]
        child_sub = sub_cls[child_conn[1]]
    else:
        sub_int = int(parent_conn[1])
        parent_sub = sub_int * 2 + (parent_conn[2] == "on")
        child_sub = sub_int * 2 + (parent_conn[2] == "in")

    base = (parent, child, parent_sub, child_sub, parent_conn[3], child_conn[3])
    # reversed traversal negates yaw/slide, except when flipped: Ry(yaw) @ RX_PI (incl. slide) is self-inverse
    backward = not forward and not flip

    if family == "stud":
        return StudEdge(*base, yaw=_qdeg(yaw if forward else -yaw))
    if family == "hinge":
        return HingeEdge(*base, flip=flip, yaw=_qdeg(yaw if forward or flip else -yaw))
    if family == "axle":
        return AxleEdge(
            *base,
            flip=flip,
            yaw=_qdeg(-rot[0] if backward else rot[0]),
            slide=round(-yaw if backward else yaw),
        )
    if family == "ball":
        return BallEdge(*base, rx=_qdeg(rot[0]), ry=_qdeg(rot[1]), rz=_qdeg(rot[2]))
    return FixedEdge(*base)


# parallel-edge tie-break: walks prefer the most parameter-expressive family
_FAM_RANK = {"axle": 0, "ball": 1, "hinge": 2, "stud": 3, "fixed": 4}


def _adjacency(edges, nodes=None):
    """Adjacency lists in canonical order (neighbor, family expressiveness, connector ids), making
    every walk a function of the edge set rather than the array order."""
    fams, a_conns, b_conns = edges["family"].tolist(), edges["a_conn"].tolist(), edges["b_conn"].tolist()
    adj = {}
    for i, (a, b) in enumerate(zip(edges["a"].tolist(), edges["b"].tolist())):
        if nodes is not None and a not in nodes:
            continue
        adj.setdefault(a, []).append((b, i, True))
        adj.setdefault(b, []).append((a, i, False))
    for lst in adj.values():
        lst.sort(key=lambda t: (t[0], _FAM_RANK[fams[t[1]]], a_conns[t[1]], b_conns[t[1]]))
    return adj


def _component_adjacency(graph: Graph, component: int):
    if component >= len(graph.components):
        raise ValueError(f"no component {component} (graph has {len(graph.components)})")
    comp = set(graph.components[component])
    return comp, _adjacency(graph.edges, comp)


def _build_tree(graph: Graph, order, parent_of) -> Tree:
    """Assemble a walk's Tree. order: graph node indices in visit order (tree index = position).
    parent_of: node -> (parent node, edge index, fwd) -- the exact edge the walk accepted."""
    remap = {old: new for new, old in enumerate(order)}
    parts = tuple(Part(graph.part_ids[o], graph.colors[o]) for o in order)
    tree_edges = tuple(
        _make_tree_edge(graph.edges[ei], remap[par], remap[node], fwd, graph.part_ids)
        for node in order
        if node in parent_of
        for par, ei, fwd in [parent_of[node]]
    )
    return Tree(parts, tree_edges)


def sample_tree(
    graph: Graph,
    component: int = 0,
    *,
    method: str = "random",
    seed: int | None = None,
) -> Tree:
    comp, adj = _component_adjacency(graph, component)

    parent_of = {}
    if method == "bfs":
        root = seed if seed is not None else min(comp)
        if root not in comp:
            raise ValueError(f"bfs seed {root} not in component {component}")
        order = []
        visited = {root}
        queue = [root]
        while queue:
            u = queue.pop(0)
            order.append(u)
            for v, ei, fwd in adj.get(u, []):
                if v not in visited:
                    visited.add(v)
                    parent_of[v] = (u, ei, fwd)
                    queue.append(v)
    else:
        rng = np.random.default_rng(seed)
        root = sorted(comp)[rng.integers(len(comp))]
        order = [root]
        visited = {root}
        stack = [root]
        while stack:
            nbrs = [(v, ei, fwd) for v, ei, fwd in adj.get(stack[-1], []) if v not in visited]
            if not nbrs:
                stack.pop()
                continue
            v, ei, fwd = nbrs[rng.integers(len(nbrs))]
            visited.add(v)
            parent_of[v] = (stack[-1], ei, fwd)
            order.append(v)
            stack.append(v)

    return _build_tree(graph, order, parent_of)


def sample_collision_free_tree(
    graph: Graph,
    component: int = 0,
    *,
    seed: int | None = None,
    max_parts: int = 100,
) -> Tree:
    """Random frontier walk over one component, rejecting any part that collides with the placed
    scene (fixed-edge children exempt their parent, which they intentionally overlap). Placements
    are decoded from the quantized tree edges, so the emitted tree realizes to exactly the
    geometry that was collision-checked. The walk may cover only part of the component."""
    from . import collision  # deferred: the meshlib import is heavy

    comp, adj = _component_adjacency(graph, component)
    edges = graph.edges
    rng = np.random.default_rng(seed)

    def step(ei, fwd):
        """Child transform relative to the parent, snapped to the quantized tree edge."""
        row = edges[ei]
        p, c = (int(row["a"]), int(row["b"])) if fwd else (int(row["b"]), int(row["a"]))
        e = _make_tree_edge(row, 0, 1, fwd, graph.part_ids)
        r2 = np.array([_tree_edge_row(e, 0, 1, graph.part_ids[p], graph.part_ids[c])], dtype=_EDGE_DT)[0]
        M = _edge_matrix(
            r2,
            graph.part_ids[p] if r2["a"] == 0 else graph.part_ids[c],
            graph.part_ids[c] if r2["a"] == 0 else graph.part_ids[p],
        )
        return M if r2["a"] == 0 else np.linalg.inv(M)

    start = sorted(comp)[rng.integers(len(comp))]
    order = [start]
    parent_of = {}
    placed = {start: 0}
    mats = [np.eye(4)]
    scene = collision.CollisionScene()
    scene.add(graph.part_ids[start], mats[0])
    frontier = [(ei, start, v, fwd) for v, ei, fwd in adj.get(start, [])]

    while len(order) < max_parts and frontier:
        ei, parent, child, fwd = frontier.pop(int(rng.integers(len(frontier))))
        if child in placed:
            continue
        mat = mats[placed[parent]] @ step(ei, fwd)
        pid = graph.part_ids[child]
        if str(edges[ei]["family"]) == "fixed":
            if any(c != placed[parent] for c in scene.check(pid, mat)):
                continue
        elif scene.check(pid, mat, first_only=True):
            continue
        placed[child] = len(order)
        parent_of[child] = (parent, ei, fwd)
        order.append(child)
        mats.append(mat)
        scene.add(pid, mat)
        frontier.extend((ei2, child, v2, fwd2) for v2, ei2, fwd2 in adj.get(child, []) if v2 not in placed)

    return _build_tree(graph, order, parent_of)


def _edge_matrix(row, a_pid: int, b_pid: int) -> np.ndarray:
    """Relative transform of part b in part a's frame for one structured edge row."""
    fa = _part_info(a_pid).frames[int(row["a_conn"])]
    fb = _part_info(b_pid).frames[int(row["b_conn"])]
    return fa @ _graph_edge_rotation(row) @ np.linalg.inv(fb)


def _graph_edge_rotation(edge) -> np.ndarray:
    """4x4 edge transform in the stored a->b direction; continuous (float) params, not quantized."""
    fam = str(edge["family"])
    if fam == "stud":
        return ry4(edge["yaw"])
    if fam == "hinge":
        R = ry4(edge["yaw"])
        return R @ RX_PI if edge["flip"] else R
    if fam == "axle":
        R = ry4(edge["rot"][0])
        if edge["flip"]:
            R = R @ RX_PI
        R[1, 3] = -edge["yaw"]
        return R
    if fam == "ball":
        R = np.eye(4)
        R[:3, :3] = Rotation.from_euler("xyz", edge["rot"]).as_matrix()
        return R
    if fam == "fixed":
        return np.eye(4)
    raise ValueError(f"unknown edge family: {fam!r}")


def decode_graph(graph: Graph) -> list[np.ndarray]:
    """Absolute per-part transforms from continuous (unquantized) edge params; roots each component at
    identity. Cycles are over-constrained, so realization is not unique; this returns the canonical
    spanning walk (BFS from the lowest part index)."""
    transforms = [np.eye(4) for _ in range(len(graph.part_ids))]
    edges = graph.edges
    adj = _adjacency(edges)
    for comp in graph.components:
        root = min(comp)
        seen, queue = {root}, [root]
        while queue:
            u = queue.pop(0)
            for v, i, fwd in adj.get(u, []):
                if v in seen:
                    continue
                seen.add(v)
                M = _edge_matrix(edges[i], graph.part_ids[int(edges[i]["a"])], graph.part_ids[int(edges[i]["b"])])
                transforms[v] = transforms[u] @ (M if fwd else np.linalg.inv(M))
                queue.append(v)
    return transforms


def _tree_edge_row(e, parent: int, child: int, parent_pid: int, child_pid: int):
    """Dequantize a tree edge into a structured edge row (the only quantized -> continuous map).
    Ball edges come out canonical a=in-side; all others a=parent."""
    if isinstance(e, (StudEdge, AxleEdge)):
        sub_cls = StudSub if isinstance(e, StudEdge) else AxleSub
        ps, pp = sub_cls(e.parent_sub).name, None
        cs, cp = sub_cls(e.child_sub).name, None
    else:
        sub_int = e.parent_sub // 2
        ps, pp = str(sub_int), "in" if e.parent_sub % 2 == 0 else "on"
        cs, cp = str(sub_int), "on" if pp == "in" else "in"
    a_conn = _part_info(parent_pid).flat_of[(ps, pp, e.parent_conn)]
    b_conn = _part_info(child_pid).flat_of[(cs, cp, e.child_conn)]

    if isinstance(e, StudEdge):
        fam, yaw, flip, rot = "stud", math.radians(e.yaw), False, (0.0, 0.0, 0.0)
    elif isinstance(e, HingeEdge):
        fam, yaw, flip, rot = "hinge", math.radians(e.yaw), e.flip, (0.0, 0.0, 0.0)
    elif isinstance(e, AxleEdge):
        fam, yaw, flip, rot = "axle", float(e.slide), e.flip, (math.radians(e.yaw), 0.0, 0.0)
    elif isinstance(e, BallEdge):
        fam, yaw, flip, rot = "ball", 0.0, False, (math.radians(e.rx), math.radians(e.ry), math.radians(e.rz))
    else:
        fam, yaw, flip, rot = "fixed", 0.0, False, (0.0, 0.0, 0.0)

    a, b, ac, bc = parent, child, a_conn, b_conn
    if fam == "ball" and e.parent_sub % 2:  # canonical a=in-side; swap when the parent is on-side
        a, b, ac, bc = b, a, bc, ac
    return (a, b, ac, bc, fam, yaw, flip, rot)


def tree_to_graph(tree: Tree) -> Graph:
    """Builds a rootless Graph: edges are dequantized to float; ball edges are stored canonical
    a=in-side, all others parent->child. decode_graph realizes the geometry."""
    part_ids = tuple(p.part_id for p in tree.parts)
    colors = tuple(p.color for p in tree.parts)

    edges = np.empty(len(tree.edges), dtype=_EDGE_DT)
    for i, e in enumerate(tree.edges):
        edges[i] = _tree_edge_row(e, e.parent, i + 1, part_ids[e.parent], part_ids[i + 1])

    components = _components_from_edges(len(tree.parts), edges["a"], edges["b"])
    return Graph(part_ids, colors, None, edges, components)


def graph_to_ldr(graph: Graph) -> str:
    """Uses stored transforms, or decode_graph when rootless. A rootless graph must be
    single-component (separate components share no frame and decode onto the origin), and the
    written geometry is the canonical realization -- one spanning solution when cycles are
    over-constrained."""
    if graph.transforms is None and len(graph.components) > 1:
        raise ValueError(f"rootless graph has {len(graph.components)} components with no shared frame")
    transforms = graph.transforms if graph.transforms is not None else decode_graph(graph)
    stems = load_catalog().id_to_stem
    lines = []
    for part_id, color, mat in zip(graph.part_ids, graph.colors, transforms):
        nums = " ".join(f"{round(v, 6) + 0.0:.12g}" for v in [*mat[:3, 3], *mat[:3, :3].flat])
        lines.append(f"1 {color} {nums} {stems[part_id]}.dat")
    return "\n".join(lines) + "\n"


# part_ids int32 (vocab index), colors int32 (LDR code), edge_kind uint8 (Kind value), edge_idx int32 rows
# [a,a_conn,b,b_conn], edge_yaw float, edge_rot float (n_edges, 3) xyz, edge_flip bool, node_ptr/edge_ptr int64.
# Optional transforms (total_nodes, 4, 4) float, CSR by node_ptr: present iff the batch is rooted.
_NPZ_KEYS = ("part_ids", "colors", "edge_kind", "edge_idx", "edge_yaw", "edge_flip", "edge_rot")


def _components_from_edges(n, a, b):
    """Connected components, largest first."""
    neighbors = {}
    for u, v in zip(a.tolist(), b.tolist()):
        neighbors.setdefault(u, []).append(v)
        neighbors.setdefault(v, []).append(u)
    visited = set()
    components = []
    for start in range(n):
        if start in visited:
            continue
        component, stack = [], [start]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                component.append(node)
                stack.extend(neighbors.get(node, []))
        components.append(component)
    components.sort(key=len, reverse=True)
    return tuple(tuple(c) for c in components)


def _graph_from_arrays(pids, colors, kind, idx, yaw, flip, rot, mats) -> Graph:
    part_ids = tuple(int(i) for i in pids)
    color_ints = tuple(int(c) for c in colors)
    edges = np.empty(len(kind), dtype=_EDGE_DT)
    edges["a"], edges["a_conn"], edges["b"], edges["b_conn"] = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
    edges["family"] = np.array(_FAM_NAMES)[kind]
    edges["yaw"], edges["flip"], edges["rot"] = yaw, flip, rot
    return Graph(part_ids, color_ints, mats, edges, _components_from_edges(len(part_ids), edges["a"], edges["b"]))


def load_graphs(path) -> list[Graph]:
    """CSR-batched .npz: node_ptr/edge_ptr offsets split one concatenated array per field. Per-part
    transforms are restored if the file carries them, else the graphs are rootless."""
    with np.load(path, allow_pickle=False) as d:
        pids, colors, kind, idx, yaw, flip, rot = (d[k] for k in _NPZ_KEYS)
        n_off, e_off = d["node_ptr"][1:-1], d["edge_ptr"][1:-1]
        mats = np.split(d["transforms"], n_off) if "transforms" in d.files else [None] * (len(d["node_ptr"]) - 1)
        cols = (
            np.split(pids, n_off),
            np.split(colors, n_off),
            np.split(kind, e_off),
            np.split(idx, e_off),
            np.split(yaw, e_off),
            np.split(flip, e_off),
            np.split(rot, e_off),
            mats,
        )
        return [_graph_from_arrays(*g) for g in zip(*cols)]


def save_graphs(graphs, path) -> None:
    """Write Graphs to a CSR-batched .npz. Per-part transforms are stored iff every graph carries them
    (so the load is rooted); a batch mixing rooted and rootless graphs is rejected."""
    rooted = [g.transforms is not None for g in graphs]
    if any(rooted) and not all(rooted):
        raise ValueError("cannot save a batch mixing rooted and rootless graphs")
    part_ids, colors, node_ptr, edge_ptr = [], [], [0], [0]
    for g in graphs:
        part_ids.extend(g.part_ids)
        colors.extend(g.colors)
        node_ptr.append(len(part_ids))
        edge_ptr.append(edge_ptr[-1] + len(g.edges))
    e = np.concatenate([g.edges for g in graphs]) if graphs else np.empty(0, dtype=_EDGE_DT)
    arrays = dict(
        part_ids=np.array(part_ids, dtype=np.int32),
        colors=np.array(colors, dtype=np.int32),
        node_ptr=np.array(node_ptr, dtype=np.int64),
        edge_ptr=np.array(edge_ptr, dtype=np.int64),
        edge_kind=np.array([Kind[f] for f in e["family"].tolist()], dtype=np.uint8),
        edge_idx=np.column_stack([e["a"], e["a_conn"], e["b"], e["b_conn"]]),
        edge_yaw=e["yaw"],
        edge_flip=e["flip"],
        edge_rot=e["rot"],
    )
    if graphs and all(rooted):
        arrays["transforms"] = np.concatenate([g.transforms for g in graphs])
    np.savez_compressed(path, **arrays)
