"""Order-independent, optimization-training-only reference graph."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from .edge_contract import PYG_EDGE_CONTRACT


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().numpy().tobytes()


@dataclass(frozen=True)
class ReferenceGraphState:
    subject_ids: tuple[str, ...]
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    edge_features: torch.Tensor
    graph_sha256: str
    fit_partition: str = "optimization_train_only"

    def clone(self) -> "ReferenceGraphState":
        return ReferenceGraphState(
            self.subject_ids,
            self.edge_index.clone(),
            self.edge_attr.clone(),
            self.edge_features.clone(),
            self.graph_sha256,
            self.fit_partition,
        )


def _hash_graph(ids: tuple[str, ...], edge_index: torch.Tensor, edge_attr: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update("\n".join(ids).encode())
    digest.update(_tensor_bytes(edge_index))
    digest.update(_tensor_bytes(edge_attr))
    return digest.hexdigest()


def build_reference_graph(
    subject_ids: tuple[str, ...],
    edge_features: torch.Tensor,
    *,
    optimization_train_ids: tuple[str, ...],
) -> ReferenceGraphState:
    """Build a directed complete reference graph without SITE_ID.

    ``edge_features`` are explicit allowed numeric covariates (for example age
    and sex); callers cannot pass a site field through this API.
    """
    if len(subject_ids) != len(set(subject_ids)) or edge_features.shape[0] != len(subject_ids):
        raise ValueError("reference subject IDs must be unique and aligned")
    if set(subject_ids) != set(optimization_train_ids):
        raise ValueError("reference graph must contain optimization-training subjects only")
    order = sorted(range(len(subject_ids)), key=lambda i: subject_ids[i])
    ids = tuple(subject_ids[i] for i in order)
    features = edge_features.detach().clone()[order]
    n = len(ids)
    pairs = [(source, target) for source in range(n) for target in range(n) if source != target]
    edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous() if pairs else torch.empty((2, 0), dtype=torch.long)
    if pairs:
        distance = torch.linalg.vector_norm(features[edge_index[0]] - features[edge_index[1]], dim=1)
        edge_attr = torch.exp(-distance).unsqueeze(1)
    else:
        edge_attr = torch.empty((0, 1), dtype=features.dtype)
    PYG_EDGE_CONTRACT.validate(edge_index, node_count=n)
    return ReferenceGraphState(ids, edge_index, edge_attr, features, _hash_graph(ids, edge_index, edge_attr))
