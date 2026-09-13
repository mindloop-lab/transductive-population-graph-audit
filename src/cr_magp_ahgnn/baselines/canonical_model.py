"""Clean-room neural core for protocol-corrected canonical baselines."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CanonicalModelConfig:
    name: str
    input_dim: int = 64
    hidden_dim: int = 16
    layers: int = 1
    chebyshev_order: int = 3
    dropout: float = 0.3
    edge_hidden: int = 8
    learned_edges: bool = False
    residual: bool = False
    layer_concat: bool = False
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    max_epochs: int = 120
    patience: int = 25


class SpectralLayer(nn.Module):
    """Dense Chebyshev message passing with explicit directed propagation."""

    def __init__(self, in_dim: int, out_dim: int, order: int):
        super().__init__()
        if order < 1:
            raise ValueError("Chebyshev order must be positive")
        self.order = order
        self.linear = nn.Linear(in_dim * (order + 1), out_dim)

    def forward(self, x: torch.Tensor, propagation: torch.Tensor) -> torch.Tensor:
        terms = [x]
        if self.order >= 1:
            terms.append(propagation @ x)
        for _ in range(2, self.order + 1):
            terms.append(2.0 * propagation @ terms[-1] - terms[-2])
        return self.linear(torch.cat(terms, dim=-1))


class CanonicalPopulationModel(nn.Module):
    """Common independently implemented core parameterized per cited baseline."""

    def __init__(self, config: CanonicalModelConfig):
        super().__init__()
        self.config = config
        self.edge_network = (
            nn.Sequential(
                nn.Linear(4, config.edge_hidden), nn.ReLU(),
                nn.Linear(config.edge_hidden, 1), nn.Sigmoid(),
            )
            if config.learned_edges else None
        )
        dimensions = [config.input_dim] + [config.hidden_dim] * config.layers
        self.layers = nn.ModuleList(
            SpectralLayer(dimensions[i], dimensions[i + 1], config.chebyshev_order)
            for i in range(config.layers)
        )
        classifier_dim = config.hidden_dim * (config.layers if config.layer_concat else 1)
        self.classifier = nn.Linear(classifier_dim, 2)

    def _propagation(
        self,
        node_count: int,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_features: torch.Tensor | None,
    ) -> torch.Tensor:
        weights = edge_weight
        if self.edge_network is not None:
            if edge_features is None or edge_features.shape[-1] != 4:
                raise ValueError("learned-edge model requires four edge features")
            weights = weights * self.edge_network(edge_features).squeeze(-1)
        matrix = torch.zeros((node_count, node_count), dtype=weights.dtype, device=weights.device)
        if edge_index.numel():
            source, target = edge_index
            matrix.index_put_((target, source), weights, accumulate=True)
        matrix = matrix + torch.eye(node_count, dtype=weights.dtype, device=weights.device)
        return matrix / matrix.sum(dim=1, keepdim=True).clamp_min(1e-12)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        propagation = self._propagation(len(x), edge_index, edge_weight, edge_features)
        hidden = x
        outputs: list[torch.Tensor] = []
        for layer in self.layers:
            updated = F.relu(layer(hidden, propagation))
            if self.config.residual and updated.shape == hidden.shape:
                updated = updated + hidden
            hidden = F.dropout(updated, p=self.config.dropout, training=self.training)
            outputs.append(hidden)
        embedding = torch.cat(outputs, dim=-1) if self.config.layer_concat else outputs[-1]
        return self.classifier(embedding), embedding

    @staticmethod
    def loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels)

    def optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(), lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def save_checkpoint(self, path: Path, *, epoch: int, validation_score: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"config": asdict(self.config), "epoch": epoch,
             "validation_score": validation_score, "state_dict": self.state_dict()}, path,
        )

    @classmethod
    def strict_reload(cls, path: Path, *, map_location: str | torch.device = "cpu"):
        payload = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(CanonicalModelConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"], strict=True)
        return model, payload


def prediction_rows(
    subject_ids: tuple[str, ...], indices: np.ndarray, labels: np.ndarray,
    logits: torch.Tensor, embeddings: torch.Tensor, threshold: float,
) -> list[dict[str, object]]:
    probability = torch.softmax(logits.detach().cpu(), dim=-1)[:, 1].numpy()
    vectors = embeddings.detach().cpu().numpy()
    return [
        {"subject_id": subject_ids[int(index)], "subject_index": int(index),
         "label": int(labels[int(index)]), "probability": float(probability[position]),
         "prediction": int(probability[position] >= threshold),
         "embedding": vectors[position].astype(float).tolist()}
        for position, index in enumerate(indices)
    ]


def config_sha256(config: CanonicalModelConfig) -> str:
    encoded = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
