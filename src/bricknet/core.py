"""Shared types: parts, edges, trees, graphs, and the part catalog."""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

# unordered pairs; elements stored lexicographically sorted because membership is tested as (min, max)
_STUD_MATES = frozenset({("hole", "open"), ("hole", "stud"), ("open", "post"), ("open", "tube"), ("stud", "tube")})
_AXLE_MATES = frozenset({("axle", "cross"), ("axle", "socket"), ("bar", "clip"), ("bar", "cross"), ("pin", "socket")})


class StudSub(IntEnum):
    stud = 0
    open = 1
    hole = 2
    tube = 3
    post = 4


class AxleSub(IntEnum):
    bar = 0
    clip = 1
    axle = 2
    cross = 3
    pin = 4
    socket = 5


@dataclass(frozen=True)
class Part:
    part_id: int  # vocabulary index, file order
    color: int


@dataclass(frozen=True)
class Edge:
    """parent_sub/child_sub: StudEdge/AxleEdge store a StudSub/AxleSub member;
    HingeEdge/BallEdge/FixedEdge store subtype * 2 + polarity_bit (0 = "in", 1 = "on")."""

    parent: int
    child: int
    parent_sub: int
    child_sub: int
    parent_conn: int
    child_conn: int


@dataclass(frozen=True)
class StudEdge(Edge):
    yaw: int = 0


@dataclass(frozen=True)
class HingeEdge(Edge):
    flip: bool = False
    yaw: int = 0


@dataclass(frozen=True)
class AxleEdge(Edge):
    flip: bool = False
    yaw: int = 0
    slide: int = 0


@dataclass(frozen=True)
class BallEdge(Edge):
    rx: int = 0
    ry: int = 0
    rz: int = 0


@dataclass(frozen=True)
class FixedEdge(Edge):
    pass


@dataclass(frozen=True)
class Tree:
    parts: tuple[Part, ...]
    edges: tuple[Edge, ...]


@dataclass(frozen=True)
class Graph:
    part_ids: tuple[int, ...]
    colors: tuple[int, ...]
    transforms: np.ndarray | None  # (n, 4, 4) absolute; None when rootless
    edges: np.ndarray  # structured array
    components: tuple[tuple[int, ...], ...]  # largest first


@dataclass(frozen=True)
class Catalog:
    conn_counts: dict[int, dict[tuple[str, str | None], int]]  # keyed by part_id
    id_to_stem: tuple[str, ...]  # lowercase ".dat" basenames, extension stripped
    stem_to_id: dict[str, int]
    id_to_name: tuple[str, ...]
    name_to_id: dict[str, int]
    color_to_code: dict[str, int]
    code_to_color: dict[int, str]
