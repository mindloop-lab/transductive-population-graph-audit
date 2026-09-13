"""Population messages with an explicit one-way query operation."""

from __future__ import annotations

import torch
from torch import nn


class PopulationEncoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.reference_update = nn.Linear(embedding_dim, embedding_dim)
        self.query_update = nn.Linear(embedding_dim * 2, embedding_dim)

    def encode_reference(self, embeddings: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        aggregate = torch.zeros_like(embeddings)
        weights = torch.zeros((embeddings.shape[0], 1), dtype=embeddings.dtype, device=embeddings.device)
        if edge_index.numel():
            source, target = edge_index
            aggregate.index_add_(0, target, embeddings[source] * edge_attr)
            weights.index_add_(0, target, edge_attr)
        message = aggregate / weights.clamp_min(1e-12)
        return embeddings + torch.relu(self.reference_update(message))

    def encode_query(self, query: torch.Tensor, references: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        weights = edge_attr / edge_attr.sum().clamp_min(1e-12)
        message = (references * weights).sum(dim=0, keepdim=True)
        return query + torch.relu(self.query_update(torch.cat((query, message), dim=1)))
