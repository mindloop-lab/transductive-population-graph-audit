"""PA-007 strict multi-atlas, cross-site SupCon and label-aware CR-v2 contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.preprocessing import StandardScaler
from torch import nn

from cr_magp_ahgnn.config.schema import EXPECTED_SPLIT_SHA256


@dataclass(frozen=True)
class Phase2BConfig:
    seed: int
    folds: tuple[int, ...]
    split_sha256: str
    atlas_input_contract: dict
    atlas_hidden: int
    embedding_dim: int
    dropout: float
    learning_rate: float
    weight_decay: float
    max_epochs: int
    patience: int
    minimum_delta: float
    batch_size: int
    lambda_supcon: tuple[float, ...]
    supcon_temperature: float
    representation_k: tuple[int, ...]
    representation_gate_k: int
    representation_gate_homophily: float
    cr_k: int
    cr_temperature: float
    cr_gate_bias_init: float
    cr_gamma_init: float
    selection_metric: str
    threshold_source: str
    test_evaluations_per_model_fold: int
    formal_sota_claim_allowed: bool
    external_data_locked: bool
    phase: str
    scope: str

    @classmethod
    def load(cls, path: Path) -> "Phase2BConfig":
        values = json.loads(path.read_text())
        for name in ("folds", "lambda_supcon", "representation_k"):
            values[name] = tuple(values[name])
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        exact = {
            "seed": 666,
            "folds": (0, 1, 2),
            "split_sha256": EXPECTED_SPLIT_SHA256,
            "lambda_supcon": (0.05, 0.1, 0.2),
            "supcon_temperature": 0.1,
            "representation_k": (4, 8, 16),
            "representation_gate_k": 8,
            "representation_gate_homophily": 0.55,
            "cr_gamma_init": 0.0,
            "selection_metric": "validation_auc_plus_balanced_accuracy",
            "threshold_source": "validation_only",
            "test_evaluations_per_model_fold": 1,
            "formal_sota_claim_allowed": False,
            "external_data_locked": True,
            "phase": "2B",
            "scope": "metric_reference_development_pilot",
        }
        for name, expected in exact.items():
            if getattr(self, name) != expected:
                raise ValueError(f"PA-007 locked configuration mismatch: {name}")
        if self.atlas_input_contract != {
            "aal_columns": [0, 121],
            "cc200_columns": [121, 131],
            "ho_columns": [131, 141],
        }:
            raise ValueError("PA-007 atlas boundary mismatch")


class AtlasPreprocessor:
    """Three independent optimization-training-only StandardScalers."""

    def __init__(self):
        self.scalers = {name: StandardScaler() for name in ("aal", "cc200", "ho")}
        self.fit_indices: tuple[int, ...] | None = None

    def fit(
        self, values: dict[str, np.ndarray], indices: tuple[int, ...]
    ) -> "AtlasPreprocessor":
        self.fit_indices = tuple(indices)
        for name, scaler in self.scalers.items():
            scaler.fit(values[name][list(indices)])
        return self

    def transform(
        self, values: dict[str, np.ndarray], indices: tuple[int, ...]
    ) -> dict[str, torch.Tensor]:
        if self.fit_indices is None:
            raise RuntimeError("atlas preprocessing has not been fitted")
        return {
            name: torch.from_numpy(
                scaler.transform(values[name][list(indices)]).astype("float32")
            )
            for name, scaler in self.scalers.items()
        }


def atlas_arrays(graphs: list[dict]) -> dict[str, np.ndarray]:
    return {
        "aal": np.stack([graph["x"][:, :121].numpy().reshape(-1) for graph in graphs]),
        "cc200": np.stack([graph["x"][:, 121:131].mean(0).numpy() for graph in graphs]),
        "ho": np.stack([graph["x"][:, 131:141].mean(0).numpy() for graph in graphs]),
    }


class MultiAtlasBackbone(nn.Module):
    def __init__(
        self,
        atlas_dimensions: dict[str, int],
        hidden: int,
        embedding_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.encoders = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dimension, hidden), nn.ReLU(), nn.LayerNorm(hidden)
                )
                for name, dimension in atlas_dimensions.items()
            }
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 3, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, 2)

    def forward(
        self, values: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = [self.encoders[name](values[name]) for name in ("aal", "cc200", "ho")]
        embedding = self.fusion(torch.cat(encoded, dim=1))
        return embedding, self.classifier(embedding)


def cross_site_supcon_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    sites: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    values = functional.normalize(embeddings, dim=1)
    logits = values @ values.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(len(values), dtype=torch.bool, device=values.device)
    same_label = labels[:, None].eq(labels[None, :]) & ~self_mask
    cross_site = ~sites[:, None].eq(sites[None, :])
    preferred = same_label & cross_site
    positive = torch.where(preferred.any(dim=1, keepdim=True), preferred, same_label)
    valid = positive.any(dim=1)
    if not valid.any():
        return embeddings.sum() * 0.0
    denominator = (
        torch.exp(logits).masked_fill(self_mask, 0).sum(dim=1).clamp_min(1e-12)
    )
    log_probability = logits - denominator.log()[:, None]
    return -(
        log_probability.masked_fill(~positive, 0).sum(dim=1)
        / positive.sum(dim=1).clamp_min(1)
    )[valid].mean()


def balanced_site_batches(
    labels: torch.Tensor, sites: torch.Tensor, batch_size: int, seed: int
) -> list[torch.Tensor]:
    generator = np.random.default_rng(seed)
    groups = {}
    for index, (label, site) in enumerate(
        zip(labels.tolist(), sites.tolist(), strict=True)
    ):
        groups.setdefault((label, site), []).append(index)
    for values in groups.values():
        generator.shuffle(values)
    order = []
    while any(groups.values()):
        for key in sorted(groups):
            if groups[key]:
                order.append(groups[key].pop())
    return [
        torch.tensor(order[start : start + batch_size])
        for start in range(0, len(order), batch_size)
    ]


class LabelAwareCRv2(nn.Module):
    """Train-label-only evidence with a zero-initialized gated logit residual."""

    def __init__(self, embedding_dim: int, gate_bias: float = -2.0):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim + 2, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        self.gamma = nn.Parameter(torch.tensor(0.0))
        with torch.no_grad():
            self.gate[-1].bias.fill_(gate_bias)

    def forward(
        self,
        query_embedding: torch.Tensor,
        base_logits: torch.Tensor,
        reference_embeddings: torch.Tensor,
        reference_labels: torch.Tensor,
        *,
        k: int,
        temperature: float,
        self_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        similarity = (
            functional.normalize(query_embedding, dim=1)
            @ functional.normalize(reference_embeddings, dim=1).T
        )
        if self_positions is not None:
            similarity[torch.arange(len(query_embedding)), self_positions] = -torch.inf
        values, indices = torch.topk(similarity, k, dim=1)
        weights = torch.softmax(values / temperature, dim=1)
        neighbor_labels = reference_labels[indices]
        vote_one = (weights * neighbor_labels.float()).sum(dim=1)
        evidence = torch.stack((1.0 - vote_one, vote_one), dim=1).clamp_min(1e-6).log()
        margin = torch.abs(vote_one - 0.5) * 2.0
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=1)
        gate = torch.sigmoid(
            self.gate(
                torch.cat((query_embedding, margin[:, None], entropy[:, None]), dim=1)
            )
        )
        logits = base_logits + self.gamma * gate * (evidence - base_logits)
        return logits, {
            "gate": gate[:, 0],
            "entropy": entropy,
            "vote_margin": margin,
            "indices": indices,
        }
