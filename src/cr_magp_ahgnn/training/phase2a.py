"""Fail-closed helpers for the bounded PA-006 representation diagnostic."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cr_magp_ahgnn.config.schema import EXPECTED_SPLIT_SHA256


@dataclass(frozen=True)
class Phase2AConfig:
    seed: int
    folds: tuple[int, ...]
    split_sha256: str
    probe_models: tuple[str, ...]
    neighborhood_k: tuple[int, ...]
    linear_c: float
    mlp_hidden: int
    gcn_hidden: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    patience: int
    minimum_delta: float
    selection_metric: str
    threshold_source: str
    test_evaluations_per_candidate_fold: int
    decision_auc_threshold: float
    homophily_sufficient_threshold: float
    formal_sota_claim_allowed: bool
    external_data_locked: bool
    phase: str
    scope: str

    @classmethod
    def load(cls, path: Path) -> "Phase2AConfig":
        values = json.loads(path.read_text(encoding="utf-8"))
        values["folds"] = tuple(values["folds"])
        values["probe_models"] = tuple(values["probe_models"])
        values["neighborhood_k"] = tuple(values["neighborhood_k"])
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        exact = {
            "seed": 666,
            "folds": (0, 1, 2),
            "split_sha256": EXPECTED_SPLIT_SHA256,
            "probe_models": ("logistic_regression", "linear_svm", "two_layer_mlp"),
            "neighborhood_k": (4, 8, 16, 32),
            "test_evaluations_per_candidate_fold": 1,
            "decision_auc_threshold": 0.60,
            "formal_sota_claim_allowed": False,
            "external_data_locked": True,
            "phase": "2A",
            "scope": "representation_diagnostic_and_backbone_selection",
        }
        for name, expected in exact.items():
            if getattr(self, name) != expected:
                raise ValueError(f"PA-006 locked configuration mismatch: {name}")
        if self.selection_metric != "validation_auc_plus_balanced_accuracy":
            raise ValueError("PA-006 selection metric mismatch")
        if self.threshold_source != "validation_only":
            raise ValueError("PA-006 threshold source mismatch")


def neighbor_indices(
    query: torch.Tensor,
    reference: torch.Tensor,
    k: int,
    *,
    self_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    if k <= 0 or k > reference.shape[0] - (self_positions is not None):
        raise ValueError("invalid neighborhood size")
    distances = torch.cdist(query.float(), reference.float())
    if self_positions is not None:
        if self_positions.shape != (query.shape[0],):
            raise ValueError("self positions do not align with queries")
        distances[torch.arange(query.shape[0]), self_positions] = torch.inf
    return torch.topk(distances, k, largest=False, sorted=True).indices


def homophily_summary(
    labels: torch.Tensor,
    sites: torch.Tensor,
    query_indices: tuple[int, ...],
    reference_indices: tuple[int, ...],
    neighbors: torch.Tensor,
) -> dict:
    query_labels = labels[list(query_indices)].cpu().numpy()
    reference_labels = labels[list(reference_indices)].cpu().numpy()
    query_sites = sites[list(query_indices)].cpu().numpy()
    neighbor_labels = reference_labels[neighbors.cpu().numpy()]
    per_query = (neighbor_labels == query_labels[:, None]).mean(axis=1)
    by_class = {
        str(label): float(per_query[query_labels == label].mean())
        for label in (0, 1)
        if np.any(query_labels == label)
    }
    by_site = {
        str(int(site)): float(per_query[query_sites == site].mean())
        for site in np.unique(query_sites)
    }
    return {
        "overall": float(per_query.mean()),
        "by_class": by_class,
        "by_site": by_site,
        "query_count": len(query_indices),
    }


def attention_entropy(weights: torch.Tensor) -> torch.Tensor:
    if weights.ndim != 2 or torch.any(weights < 0):
        raise ValueError("attention weights must be a non-negative matrix")
    normalized = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return -(normalized * normalized.clamp_min(1e-12).log()).sum(dim=1)


def select_by_validation(scores: dict[str, list[float]]) -> str:
    """Select one diagnostic family without consulting any test result."""
    if not scores or any(not values for values in scores.values()):
        raise ValueError("validation score inventory is incomplete")
    if any(not np.isfinite(values).all() for values in scores.values()):
        raise ValueError("validation score inventory contains non-finite values")
    return max(sorted(scores), key=lambda name: float(np.mean(scores[name])))
