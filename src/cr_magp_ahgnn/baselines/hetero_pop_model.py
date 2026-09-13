"""Clean-room heterogeneous (same/cross-sex) population model (B representative).

Protocol-matched to the canonical SOTA baselines: same canonical 64-D fold-local
features and the same site-aware population graph contract, but the aggregation
is split into same-sex / cross-sex channels with a per-layer learnable gate.
This isolates the "heterogeneous population aggregation" variable against the
single-channel spectral baselines. It is an FC-HGNN-CLASS representative, NOT a
claim to reproduce FC-HGNN exactly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .canonical_model import SpectralLayer


@dataclass(frozen=True)
class HeteroPopConfig:
    name: str = "Hetero-Pop-C"
    input_dim: int = 64
    hidden_dim: int = 16
    layers: int = 3
    chebyshev_order: int = 3
    dropout: float = 0.3
    residual: bool = False
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    max_epochs: int = 120
    patience: int = 25


def _propagation_matrix(node_count: int, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
    matrix = torch.zeros((node_count, node_count), dtype=edge_weight.dtype, device=edge_weight.device)
    if edge_index.numel():
        source, target = edge_index
        matrix.index_put_((target, source), edge_weight, accumulate=True)
    matrix = matrix + torch.eye(node_count, dtype=edge_weight.dtype, device=edge_weight.device)
    return matrix / matrix.sum(dim=1, keepdim=True).clamp_min(1e-12)


class HeteroPopModel(nn.Module):
    """Two-channel spectral population model with per-layer sex gates."""

    def __init__(self, config: HeteroPopConfig):
        super().__init__()
        self.config = config
        self.same = nn.ModuleList(
            SpectralLayer(config.input_dim if i == 0 else config.hidden_dim, config.hidden_dim, config.chebyshev_order)
            for i in range(config.layers)
        )
        self.cross = nn.ModuleList(
            SpectralLayer(config.input_dim if i == 0 else config.hidden_dim, config.hidden_dim, config.chebyshev_order)
            for i in range(config.layers)
        )
        self.gates = nn.Parameter(torch.full((config.layers, 2), 0.5))
        self.classifier = nn.Linear(config.hidden_dim, 2)

    def forward(self, x, same_ei, cross_ei, same_w, cross_w) -> tuple[torch.Tensor, torch.Tensor]:
        n = x.shape[0]
        p_same = _propagation_matrix(n, same_ei, same_w)
        p_cross = _propagation_matrix(n, cross_ei, cross_w)
        hidden = x
        for i in range(self.config.layers):
            s = self.same[i](hidden, p_same)
            c = self.cross[i](hidden, p_cross)
            g = torch.sigmoid(self.gates[i])
            updated = F.relu(g[0] * s + g[1] * c)
            if self.config.residual and updated.shape == hidden.shape:
                updated = updated + hidden
            hidden = F.dropout(updated, p=self.config.dropout, training=self.training)
        return self.classifier(hidden), hidden

    @staticmethod
    def loss(logits, labels) -> torch.Tensor:
        return F.cross_entropy(logits, labels)

    def optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

    def save_checkpoint(self, path: Path, *, epoch: int, validation_score: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.config), "epoch": epoch,
                    "validation_score": validation_score, "state_dict": self.state_dict()}, path)

    @classmethod
    def strict_reload(cls, path: Path, *, map_location: str | torch.device = "cpu"):
        payload = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(HeteroPopConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"], strict=True)
        return model, payload


def config_sha256(config: HeteroPopConfig) -> str:
    return hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()
