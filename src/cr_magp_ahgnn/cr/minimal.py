"""Minimal Phase 1B CR modules implementing the PA-004 equations only."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CROutput:
    logits: torch.Tensor
    corrected_embedding: torch.Tensor
    reference_context: torch.Tensor
    gate: torch.Tensor


class QueryReferenceAggregator(nn.Module):
    """Compute ``r_q = sum_j a_qj z_j`` with normalized affinities."""

    def forward(
        self,
        query: torch.Tensor,
        references: torch.Tensor,
        affinities: torch.Tensor,
    ) -> torch.Tensor:
        if query.ndim == 1:
            query = query.unsqueeze(0)
        if query.ndim != 2 or references.ndim not in (2, 3) or references.shape[-1] != query.shape[1]:
            raise ValueError("query and reference embeddings must have one shared feature dimension")
        reference_count = references.shape[-2]
        if reference_count == 0:
            raise ValueError("at least one reference is required")
        if affinities.ndim == 1:
            affinities = affinities.unsqueeze(0)
        if affinities.shape != (query.shape[0], reference_count):
            raise ValueError("affinities must have shape [queries, references]")
        if references.ndim == 3 and references.shape[0] != query.shape[0]:
            raise ValueError("batched references must align with queries")
        if not torch.isfinite(affinities).all() or torch.any(affinities < 0):
            raise ValueError("affinities must be finite and non-negative")
        totals = affinities.sum(dim=1, keepdim=True)
        if torch.any(totals <= 0):
            raise ValueError("each query must have positive total reference affinity")
        weights = affinities / totals
        if references.ndim == 2:
            return weights @ references
        return torch.sum(weights.unsqueeze(-1) * references, dim=1)


class ConfidenceGate(nn.Module):
    """Compute ``sigmoid(h([z_q, r_q, abs(z_q-r_q)]))``."""

    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query: torch.Tensor, reference_context: torch.Tensor) -> torch.Tensor:
        values = torch.cat((query, reference_context, torch.abs(query - reference_context)), dim=-1)
        return torch.sigmoid(self.network(values))


class ResidualCorrection(nn.Module):
    """Compute ``z_q + alpha * g_q * W(r_q-z_q)``."""

    def __init__(self, embedding_dim: int, *, alpha_init: float = 0.0):
        super().__init__()
        if alpha_init != 0.0:
            raise ValueError("PA-004 requires alpha to initialize at exactly zero")
        self.projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        query: torch.Tensor,
        reference_context: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        if gate.shape != (query.shape[0], 1):
            raise ValueError("gate must have shape [queries, 1]")
        return query + self.alpha * gate * self.projection(reference_context - query)


class CRClassifier(nn.Module):
    """Minimal CR correction followed by the approved M0-S classifier."""

    def __init__(self, m0: nn.Module, embedding_dim: int, gate_hidden_dim: int):
        super().__init__()
        self.aggregator = QueryReferenceAggregator()
        self.gate = ConfidenceGate(embedding_dim, gate_hidden_dim)
        self.correction = ResidualCorrection(embedding_dim)
        if not callable(getattr(m0, "classify", None)) or not hasattr(m0, "population_encoder"):
            raise TypeError("CRClassifier requires an M0-S-compatible model")
        self.m0 = m0

    def forward(
        self,
        query: torch.Tensor,
        references: torch.Tensor,
        affinities: torch.Tensor,
        *,
        gate_override: torch.Tensor | None = None,
    ) -> CROutput:
        if query.ndim == 1:
            query = query.unsqueeze(0)
        context = self.aggregator(query, references, affinities)
        gate = self.gate(query, context) if gate_override is None else gate_override
        if gate.device != query.device or gate.dtype != query.dtype:
            gate = gate.to(device=query.device, dtype=query.dtype)
        corrected = self.correction(query, context, gate)
        return CROutput(self.m0.classify(corrected), corrected, context, gate)
