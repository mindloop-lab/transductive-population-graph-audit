"""Isolated reference-to-query attachment."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .edge_contract import PYG_EDGE_CONTRACT
from .reference_graph import ReferenceGraphState


@dataclass(frozen=True)
class QueryAttachment:
    query_id: str
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    reference_count: int


def attach_single_query(
    reference: ReferenceGraphState,
    query_id: str,
    query_edge_features: torch.Tensor,
) -> QueryAttachment:
    if query_edge_features.ndim != 1 or query_edge_features.shape[0] != reference.edge_features.shape[1]:
        raise ValueError("query edge features do not match the frozen reference schema")
    n = len(reference.subject_ids)
    source = torch.arange(n, dtype=torch.long)
    target = torch.full((n,), n, dtype=torch.long)
    edge_index = torch.stack((source, target))
    distance = torch.linalg.vector_norm(reference.edge_features - query_edge_features.detach(), dim=1)
    edge_attr = torch.exp(-distance).unsqueeze(1)
    PYG_EDGE_CONTRACT.validate(edge_index, node_count=n + 1)
    if edge_index.numel() and (not torch.all(edge_index[0] < n) or not torch.all(edge_index[1] == n)):
        raise AssertionError("strict query edges must be reference -> query")
    return QueryAttachment(query_id, edge_index, edge_attr, n)
