"""Frozen-reference, strict single-query inference for M0-S."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cr_magp_ahgnn.graph.query_attachment import QueryAttachment, attach_single_query
from cr_magp_ahgnn.graph.reference_graph import ReferenceGraphState
from cr_magp_ahgnn.models.m0_strict import M0Strict, StrictOutput


@dataclass(frozen=True)
class FrozenReference:
    graph: ReferenceGraphState
    subject_embeddings: torch.Tensor
    population_embeddings: torch.Tensor
    logits: torch.Tensor

    def snapshot(self) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.graph.graph_sha256, self.subject_embeddings.clone(), self.population_embeddings.clone(), self.logits.clone())


class StrictSingleQueryProtocol:
    def __init__(self, model: M0Strict, reference: FrozenReference):
        self.model = model
        self.reference = reference

    @classmethod
    def freeze(
        cls,
        model: M0Strict,
        graph: ReferenceGraphState,
        subject_ids: tuple[str, ...],
        subject_embeddings: torch.Tensor,
    ) -> "StrictSingleQueryProtocol":
        if len(subject_ids) != subject_embeddings.shape[0] or set(subject_ids) != set(graph.subject_ids):
            raise ValueError("subject embeddings must exactly match the reference graph")
        positions = {subject_id: index for index, subject_id in enumerate(subject_ids)}
        canonical = subject_embeddings[[positions[subject_id] for subject_id in graph.subject_ids]]
        population = model.population_encoder.encode_reference(canonical, graph.edge_index, graph.edge_attr)
        logits = model.classify(population)
        frozen = FrozenReference(graph.clone(), canonical.clone(), population, logits)
        return cls(model, frozen)

    def predict_one(self, query_id: str, query_embedding: torch.Tensor, query_edge_features: torch.Tensor) -> tuple[StrictOutput, QueryAttachment]:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.unsqueeze(0)
        attachment = attach_single_query(self.reference.graph, query_id, query_edge_features)
        updated = self.model.population_encoder.encode_query(
            query_embedding,
            self.reference.population_embeddings,
            attachment.edge_attr,
        )
        return StrictOutput((query_id,), self.model.classify(updated)), attachment

    def predict_many(self, query_ids: tuple[str, ...], query_embeddings: torch.Tensor, query_edge_features: torch.Tensor) -> StrictOutput:
        if len(query_ids) != query_embeddings.shape[0] or len(query_ids) != query_edge_features.shape[0]:
            raise ValueError("query IDs and tensors must be aligned")
        logits = [self.predict_one(query_id, query_embeddings[i], query_edge_features[i])[0].logits for i, query_id in enumerate(query_ids)]
        return StrictOutput(query_ids, torch.cat(logits, dim=0))
