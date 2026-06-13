"""Path-text grammar::

    sample  = node "\n" (node "\n" edge "\n")*
    node    = ID " " NAME " | " COLOR
    edge    = stud | axle | hinge | ball | fixed
    stud    = ID " stud " SUB " " ID " " SUB " " ID " " DEG
    axle    = ID " axle " SUB " " ID " " SUB " " ID " " FLIP " " DEG " " INT
    hinge   = ID " hinge " ID " " POL " " ID " " ID " " FLIP " " DEG
    ball    = ID " ball " ID " " POL " " ID " " ID " " DEG " " DEG " " DEG
    fixed   = ID " fixed " ID " " POL " " ID " " ID
    ID      = [a-z]+
    DEG     = INT in [0, 360)
    INT     = canonical integer (no leading zeros, no +)
    FLIP    = "regular" | "flip"
    POL     = "in" | "on"
    SUB     = connector subtype name
    NAME    = part name from part_names.json
    COLOR   = color name from color_names.json

"""

import math
from dataclasses import dataclass

import numpy as np

from .core import (
    AxleEdge,
    AxleSub,
    BallEdge,
    Catalog,
    Edge,
    FixedEdge,
    HingeEdge,
    Part,
    Tree,
    StudEdge,
    StudSub,
    _AXLE_MATES,
    _STUD_MATES,
)
from .data import load_catalog

EDGE_ARITY = {"stud": 7, "hinge": 8, "axle": 9, "ball": 9, "fixed": 6}


class ParseError(Exception):
    pass


@dataclass(frozen=True)
class ParseResult:
    tree: Tree
    error: str | None = None


def id2i(t: str) -> int:
    """Letter ID ('a'=0, 'z'=25, 'aa'=26, ...) to int."""
    if not (t and t.isascii() and t.isalpha() and t.islower()):
        raise ParseError(f"invalid letter ID: {t!r}")
    return sum((ord(c) - 97) * 26**i for i, c in enumerate(reversed(t))) + (26 ** len(t) - 26) // 25


def i2id(n: int) -> str:
    """Int to letter ID (inverse of id2i)."""
    if n < 0:
        raise ValueError(f"negative index: {n}")
    length, offset = 1, 0
    while n >= offset + 26**length:
        offset += 26**length
        length += 1
    n -= offset
    return "".join(chr(97 + (n // 26 ** (length - 1 - i)) % 26) for i in range(length))


def _int(tok: str, label: str) -> int:
    try:
        v = int(tok)
    except ValueError:
        raise ParseError(f"{label} is not an integer: {tok!r}") from None
    if tok != str(v):
        raise ParseError(f"{label} has non-canonical form: {tok!r}")
    return v


def _deg(tok: str) -> int:
    v = _int(tok, "degree")
    if not (0 <= v < 360):
        raise ParseError(f"degree {v} not in [0, 360)")
    return v


def _flip(tok: str) -> bool:
    if tok not in ("regular", "flip"):
        raise ParseError(f"expected regular/flip, got {tok!r}")
    return tok == "flip"


def _conn(catalog: Catalog, part_id: int, subtype: str, polarity: str | None, lid: str) -> int:
    """Index within the (subtype, polarity) subgroup."""
    if (subtype, polarity) not in catalog.conn_counts[part_id]:
        raise ParseError(f"part {part_id} has no {subtype}/{polarity} connectors")
    idx = id2i(lid)
    count = catalog.conn_counts[part_id][(subtype, polarity)]
    if idx >= count:
        raise ParseError(f"{subtype}/{polarity}[{idx}] out of range for part {part_id} (has {count})")
    return idx


def parse_node(line: str, catalog: Catalog) -> Part:
    left, sep, color = line.partition(" | ")
    if not sep:
        raise ParseError(f"node line missing ' | ': {line!r}")
    node_id, _, name = left.partition(" ")
    if not name:
        raise ParseError(f"bad node: {line!r}")
    id2i(node_id)  # labels are decorative; identity is positional
    if name not in catalog.name_to_id:
        raise ParseError(f"unknown part: {name!r}")
    if color not in catalog.color_to_code:
        raise ParseError(f"unknown color: {color!r}")
    return Part(catalog.name_to_id[name], catalog.color_to_code[color])


def parse_edge(line: str, parts: list[Part], child_part_id: int, catalog: Catalog) -> Edge:
    t = line.split(" ")
    if len(t) < 2:
        raise ParseError(f"bad edge: {line!r}")
    fam = t[1]
    if fam not in EDGE_ARITY:
        raise ParseError(f"unknown edge family: {fam!r}")
    if len(t) != EDGE_ARITY[fam]:
        raise ParseError(f"{fam} edge expected {EDGE_ARITY[fam]} tokens, got {len(t)}")
    par = id2i(t[0])
    child = len(parts)
    if par >= child:
        raise ParseError(f"bad parent ref: {t[0]!r}")
    parent_id = parts[par].part_id

    if fam in ("stud", "axle"):
        ps, cs = t[2], t[4]
        pi, ci = _conn(catalog, parent_id, ps, None, t[3]), _conn(catalog, child_part_id, cs, None, t[5])
        if (min(ps, cs), max(ps, cs)) not in (_STUD_MATES if fam == "stud" else _AXLE_MATES):
            raise ParseError(f"cannot mate subtypes {ps}+{cs}")
        sub_enum = StudSub if fam == "stud" else AxleSub
        psi, csi = sub_enum[ps], sub_enum[cs]
        if fam == "stud":
            return StudEdge(par, child, psi, csi, pi, ci, yaw=_deg(t[6]))
        return AxleEdge(par, child, psi, csi, pi, ci, flip=_flip(t[6]), yaw=_deg(t[7]), slide=_int(t[8], "slide"))

    sub = id2i(t[2])
    pol = t[3]
    if pol not in ("in", "on"):
        raise ParseError(f"bad polarity: {pol!r}")
    opp = "on" if pol == "in" else "in"
    ps, cs = sub * 2 + (pol == "on"), sub * 2 + (pol == "in")
    pi, ci = _conn(catalog, parent_id, str(sub), pol, t[4]), _conn(catalog, child_part_id, str(sub), opp, t[5])
    if fam == "hinge":
        return HingeEdge(par, child, ps, cs, pi, ci, flip=_flip(t[6]), yaw=_deg(t[7]))
    if fam == "ball":
        return BallEdge(par, child, ps, cs, pi, ci, rx=_deg(t[6]), ry=_deg(t[7]), rz=_deg(t[8]))
    return FixedEdge(par, child, ps, cs, pi, ci)


def serialize_tree(tree: Tree, catalog: Catalog | None = None) -> str:
    catalog = catalog or load_catalog()
    lines = []
    for idx, part in enumerate(tree.parts):
        name = catalog.id_to_name[part.part_id]
        lines.append(f"{i2id(idx)} {name} | {catalog.code_to_color[part.color]}")
        if idx == 0:
            continue
        e = tree.edges[idx - 1]
        p, pc, cc = i2id(e.parent), i2id(e.parent_conn), i2id(e.child_conn)
        if isinstance(e, StudEdge):
            psn, csn = StudSub(e.parent_sub).name, StudSub(e.child_sub).name
            lines.append(f"{p} stud {psn} {pc} {csn} {cc} {e.yaw}")
        elif isinstance(e, AxleEdge):
            psn, csn = AxleSub(e.parent_sub).name, AxleSub(e.child_sub).name
            f = "flip" if e.flip else "regular"
            lines.append(f"{p} axle {psn} {pc} {csn} {cc} {f} {e.yaw} {e.slide}")
        else:
            sub, pol = i2id(e.parent_sub // 2), "in" if e.parent_sub % 2 == 0 else "on"
            if isinstance(e, HingeEdge):
                f = "flip" if e.flip else "regular"
                lines.append(f"{p} hinge {sub} {pol} {pc} {cc} {f} {e.yaw}")
            elif isinstance(e, BallEdge):
                lines.append(f"{p} ball {sub} {pol} {pc} {cc} {e.rx} {e.ry} {e.rz}")
            else:
                lines.append(f"{p} fixed {sub} {pol} {pc} {cc}")
    return "\n".join(lines) + "\n"


def parse_sample(text: str, catalog: Catalog | None = None) -> ParseResult:
    catalog = catalog or load_catalog()
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    if not lines:
        return ParseResult(Tree((), ()), "empty")
    parts, edges = [], []
    try:
        parts.append(parse_node(lines[0], catalog))
        for i in range(1, len(lines) - 1, 2):
            part = parse_node(lines[i], catalog)
            edges.append(parse_edge(lines[i + 1], parts, part.part_id, catalog))
            parts.append(part)
        if len(lines) % 2 == 0:
            raise ParseError(f"dangling node without edge: {lines[-1]!r}")
    except ParseError as e:
        return ParseResult(Tree(tuple(parts), tuple(edges)), str(e))
    tree = Tree(tuple(parts), tuple(edges))
    if not text.endswith("\n"):
        return ParseResult(tree, "missing trailing newline")
    return ParseResult(tree)


RX_PI = np.diag([1.0, -1.0, -1.0, 1.0])


def ry4(angle):
    """Angle in radians."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s, 0.0], [0.0, 1.0, 0.0, 0.0], [-s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]])
