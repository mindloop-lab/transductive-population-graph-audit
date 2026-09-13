"""LOO training-reference and strict single-query Phase 1B interfaces."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch import nn

from cr_magp_ahgnn.cr.minimal import CRClassifier, CROutput
from cr_magp_ahgnn.cr.reference_only import ReferenceBoundary
from cr_magp_ahgnn.graph.query_attachment import attach_single_query
from cr_magp_ahgnn.protocols.strict_single_query import FrozenReference


def _tensor_hash(digest: hashlib._Hash, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())


def module_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        _tensor_hash(digest, tensor)
    return digest.hexdigest()


def frozen_reference_sha256(reference: FrozenReference) -> str:
    digest = hashlib.sha256()
    digest.update("\n".join(reference.graph.subject_ids).encode())
    digest.update(reference.graph.graph_sha256.encode())
    for tensor in (
        reference.graph.edge_index,
        reference.graph.edge_attr,
        reference.graph.edge_features,
        reference.subject_embeddings,
        reference.population_embeddings,
        reference.logits,
    ):
        _tensor_hash(digest, tensor)
    return digest.hexdigest()


@dataclass(frozen=True)
class ReferenceCheckpointIdentity:
    reference_sha256: str
    m0_checkpoint_sha256: str


@dataclass(frozen=True)
class LOOReferenceView:
    subject_ids: tuple[str, ...]
    embeddings: torch.Tensor


class LeaveOneOutTrainingReference:
    """Create per-training-query reference views without the query itself."""

    def __init__(self, subject_ids: tuple[str, ...], embeddings: torch.Tensor):
        if len(subject_ids) != len(set(subject_ids)):
            raise ValueError("duplicate reference subject IDs are forbidden")
        if embeddings.ndim != 2 or embeddings.shape[0] != len(subject_ids):
            raise ValueError("reference IDs and embeddings must be aligned")
        self.subject_ids = subject_ids
        self.embeddings = embeddings

    def for_query(self, query_id: str) -> LOOReferenceView:
        if query_id not in self.subject_ids:
            raise ValueError("LOO training query must belong to optimization training")
        keep = [index for index, subject_id in enumerate(self.subject_ids) if subject_id != query_id]
        return LOOReferenceView(tuple(self.subject_ids[index] for index in keep), self.embeddings[keep])


class StrictCRSingleQueryProtocol:
    """Strict per-query CR inference against one immutable training reference."""

    def __init__(
        self,
        model: CRClassifier,
        reference: FrozenReference,
        boundary: ReferenceBoundary,
        *,
        expected_identity: ReferenceCheckpointIdentity,
        actual_m0_checkpoint_sha256: str,
    ):
        boundary.validate(reference.graph.subject_ids)
        if len(reference.graph.subject_ids) != len(set(reference.graph.subject_ids)):
            raise ValueError("duplicate reference subject IDs are forbidden")
        actual = ReferenceCheckpointIdentity(
            frozen_reference_sha256(reference), actual_m0_checkpoint_sha256
        )
        if actual != expected_identity:
            raise ValueError("stale reference artifact or M0 checkpoint mismatch")
        self.model = model
        self.reference = FrozenReference(
            reference.graph.clone(),
            reference.subject_embeddings.detach().clone(),
            reference.population_embeddings.detach().clone(),
            reference.logits.detach().clone(),
        )
        self.boundary = boundary
        self.identity = actual

    def predict_one(
        self,
        query_id: str,
        query_embedding: torch.Tensor,
        query_edge_features: torch.Tensor,
        *,
        gate_override: torch.Tensor | None = None,
    ) -> CROutput:
        if query_id in set(self.boundary.optimization_train_ids):
            raise ValueError("validation/test query cannot be a reference subject")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.unsqueeze(0)
        attachment = attach_single_query(self.reference.graph, query_id, query_edge_features)
        query_population = self.model.m0.population_encoder.encode_query(
            query_embedding,
            self.reference.population_embeddings,
            attachment.edge_attr,
        )
        return self.model(
            query_population,
            self.reference.population_embeddings,
            attachment.edge_attr.squeeze(1),
            gate_override=gate_override,
        )

    def predict_many(
        self,
        query_ids: tuple[str, ...],
        query_embeddings: torch.Tensor,
        query_edge_features: torch.Tensor,
    ) -> tuple[CROutput, ...]:
        if len(query_ids) != query_embeddings.shape[0] or len(query_ids) != query_edge_features.shape[0]:
            raise ValueError("query IDs and tensors must be aligned")
        return tuple(
            self.predict_one(query_id, query_embeddings[index], query_edge_features[index])
            for index, query_id in enumerate(query_ids)
        )
