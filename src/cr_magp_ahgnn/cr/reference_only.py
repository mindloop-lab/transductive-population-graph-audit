"""Reference-only Phase 1A protocol with residual computation disabled."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from cr_magp_ahgnn.config.schema import EXPECTED_SPLIT_SHA256
from cr_magp_ahgnn.models.m0_strict import StrictOutput

if TYPE_CHECKING:
    from cr_magp_ahgnn.protocols.strict_single_query import StrictSingleQueryProtocol


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


@dataclass(frozen=True)
class ReferenceBoundary:
    """Explicit partition and fit boundary for one frozen reference graph."""

    split_sha256: str
    optimization_train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    fit_statistic_ids: tuple[str, ...]

    def validate(self, reference_ids: tuple[str, ...]) -> None:
        if self.split_sha256 != EXPECTED_SPLIT_SHA256:
            raise ValueError("split SHA-256 is not PA-003-authorized")
        parts = tuple(map(set, (self.optimization_train_ids, self.validation_ids, self.test_ids)))
        if any(parts[left] & parts[right] for left in range(3) for right in range(left + 1, 3)):
            raise ValueError("optimization, validation and test partitions must be disjoint")
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError("reference IDs must be unique")
        if set(reference_ids) != parts[0]:
            raise ValueError("reference graph must contain optimization-training subjects only")
        if set(self.fit_statistic_ids) != parts[0]:
            raise ValueError("all fitted statistics must use optimization-training subjects only")


class Phase1AReferenceProtocol:
    """Thin residual-off interface over the approved M0-S protocol.

    Phase 1A deliberately contains no reliability estimator, residual branch,
    trainable gate or alternative prediction path. Both scales are immutable
    scalar zeros and every prediction delegates to M0-S.
    """

    gamma_h: float = 0.0
    gamma_e: float = 0.0

    def __init__(self, baseline: StrictSingleQueryProtocol, boundary: ReferenceBoundary):
        boundary.validate(baseline.reference.graph.subject_ids)
        self._baseline = baseline
        self.boundary = boundary

    @property
    def baseline(self) -> StrictSingleQueryProtocol:
        return self._baseline

    def predict_one(
        self,
        query_id: str,
        query_embedding: torch.Tensor,
        query_edge_features: torch.Tensor,
    ) -> StrictOutput:
        if query_id in set(self.boundary.optimization_train_ids):
            raise ValueError("a frozen reference subject cannot be inserted as a query")
        return self._baseline.predict_one(query_id, query_embedding, query_edge_features)[0]

    def predict_many(
        self,
        query_ids: tuple[str, ...],
        query_embeddings: torch.Tensor,
        query_edge_features: torch.Tensor,
    ) -> StrictOutput:
        if len(query_ids) != query_embeddings.shape[0] or len(query_ids) != query_edge_features.shape[0]:
            raise ValueError("query IDs and tensors must be aligned")
        outputs = [
            self.predict_one(query_id, query_embeddings[index], query_edge_features[index]).logits
            for index, query_id in enumerate(query_ids)
        ]
        return StrictOutput(query_ids, torch.cat(outputs, dim=0))

    def save_reference_artifact(
        self,
        output_dir: Path,
        *,
        allowed_output_root: Path,
        protected_roots: tuple[Path, ...] = (),
    ) -> dict:
        output = output_dir.resolve()
        allowed = allowed_output_root.resolve()
        if not _is_within(output, allowed):
            raise ValueError("reference artifact must be inside the explicit run root")
        if any(_is_within(output, root.resolve()) for root in protected_roots):
            raise ValueError("reference artifact output overlaps protected evidence or canonical data")

        self.boundary.validate(self._baseline.reference.graph.subject_ids)
        output.mkdir(parents=True, exist_ok=False)
        state_path = output / "reference_state.pt"
        temporary = output / f"reference_state.pt.tmp.{os.getpid()}"
        reference = self._baseline.reference
        torch.save(
            {
                "subject_ids": reference.graph.subject_ids,
                "edge_index": reference.graph.edge_index.detach().cpu(),
                "edge_attr": reference.graph.edge_attr.detach().cpu(),
                "edge_features": reference.graph.edge_features.detach().cpu(),
                "subject_embeddings": reference.subject_embeddings.detach().cpu(),
                "population_embeddings": reference.population_embeddings.detach().cpu(),
                "logits": reference.logits.detach().cpu(),
            },
            temporary,
        )
        temporary.replace(state_path)
        state_sha = _sha256(state_path)
        manifest = {
            "schema_version": 1,
            "phase": "1A",
            "protocol": "strict_single_query_reference_only",
            "training_performed": False,
            "cr_computation_enabled": False,
            "residual_scales": {"gamma_h": 0.0, "gamma_e": 0.0},
            "split_sha256": self.boundary.split_sha256,
            "fit_partition": "optimization_train_only",
            "reference_subject_ids": list(reference.graph.subject_ids),
            "reference_subject_count": len(reference.graph.subject_ids),
            "reference_graph_sha256": reference.graph.graph_sha256,
            "reference_state_file": state_path.name,
            "reference_state_sha256": state_sha,
            "query_contract": {
                "direction": "reference_to_query",
                "query_to_reference_edges": 0,
                "query_to_query_edges": 0,
                "test_test_edges": 0,
            },
        }
        manifest_path = output / "reference_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        sums_path = output / "SHA256SUMS"
        sums_path.write_text(
            f"{state_sha}  {state_path.name}\n{_sha256(manifest_path)}  {manifest_path.name}\n",
            encoding="utf-8",
        )
        return manifest
