"""Explicit graph contracts for the strict M0-S protocol."""

from .edge_contract import EdgeContract
from .query_attachment import QueryAttachment, attach_single_query
from .reference_graph import ReferenceGraphState, build_reference_graph

__all__ = [
    "EdgeContract",
    "QueryAttachment",
    "ReferenceGraphState",
    "attach_single_query",
    "build_reference_graph",
]
