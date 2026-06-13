"""BrickNet: Graph-Backed Generative Brick Assembly."""

from .core import AxleEdge, BallEdge, Edge, FixedEdge, Graph, HingeEdge, Part, StudEdge, Tree
from .data import load_catalog, load_connector_list, load_connectors
from .graph import (
    decode_graph,
    graph_to_ldr,
    load_graphs,
    parse_ldr,
    sample_collision_free_tree,
    sample_tree,
    save_graphs,
    tree_to_graph,
)
from .score import score_text
from .tree import parse_sample, serialize_tree

__version__ = "0.1.0"

__all__ = [
    "AxleEdge",
    "BallEdge",
    "Edge",
    "FixedEdge",
    "Graph",
    "HingeEdge",
    "Part",
    "StudEdge",
    "Tree",
    "decode_graph",
    "graph_to_ldr",
    "load_catalog",
    "load_connector_list",
    "load_connectors",
    "load_graphs",
    "parse_ldr",
    "parse_sample",
    "sample_collision_free_tree",
    "sample_tree",
    "save_graphs",
    "score_text",
    "serialize_tree",
    "tree_to_graph",
]
