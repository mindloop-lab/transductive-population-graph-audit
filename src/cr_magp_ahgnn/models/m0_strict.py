"""Trainable clean M0-S; Phase 0C validates it without formal training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .population_encoder import PopulationEncoder
from .subject_encoder import SubjectEncoder


@dataclass(frozen=True)
class StrictOutput:
    subject_ids: tuple[str, ...]
    logits: torch.Tensor


class M0Strict(nn.Module):
    def __init__(self, node_features: int, context_features: int, hidden: int = 32, classes: int = 2):
        super().__init__()
        self.subject_encoder = SubjectEncoder(node_features, context_features, hidden)
        self.population_encoder = PopulationEncoder(self.subject_encoder.output_dim)
        self.classifier = nn.Linear(self.subject_encoder.output_dim, classes)

    def classify(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)
