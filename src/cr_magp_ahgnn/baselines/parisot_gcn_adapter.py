"""Clean-room Parisot population-GCN data and graph adapter.

Algorithm reference: parisots/population-gcn@185e2f5d9745484b8c7994f80254740b9cbb7b75
(GPL-3.0-or-later). No upstream source is copied or imported. This module only
implements the independently stated population-graph contract on canonical data.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from cr_magp_ahgnn.config.schema import EXPECTED_SPLIT_SHA256
from cr_magp_ahgnn.data.splits import FoldBoundary, SplitManifest

from .canonical_model import CanonicalModelConfig, CanonicalPopulationModel


class CanonicalDataManager(Protocol):
    """Minimal read-only interface supplied by the CR-MAGP-AHGNN data layer."""

    subject_ids: tuple[str, ...]
    labels: np.ndarray
    sex: np.ndarray
    site: np.ndarray

    def features(self, atlas: str) -> np.ndarray: ...


@dataclass(frozen=True)
class BaselineFold:
    subject_ids: tuple[str, ...]
    features: np.ndarray
    labels: np.ndarray
    sex: np.ndarray
    site: np.ndarray
    boundary: FoldBoundary
    split_sha256: str

    @classmethod
    def from_manager(
        cls, manager: CanonicalDataManager, splits: SplitManifest, fold: int, *, atlas: str
    ) -> "BaselineFold":
        if splits.sha256 != EXPECTED_SPLIT_SHA256:
            raise ValueError("unauthorized split SHA-256")
        values = np.asarray(manager.features(atlas), dtype=np.float64)
        count = len(manager.subject_ids)
        arrays = (values, manager.labels, manager.sex, manager.site)
        if count != splits.subject_count or any(len(value) != count for value in arrays):
            raise ValueError("canonical manager subject alignment failure")
        if len(set(manager.subject_ids)) != count:
            raise ValueError("duplicate subject ID")
        return cls(
            tuple(manager.subject_ids), values, np.asarray(manager.labels),
            np.asarray(manager.sex), np.asarray(manager.site), splits.folds[fold], splits.sha256,
        )


def _correlation_similarity(features: np.ndarray) -> np.ndarray:
    distance = 1.0 - np.corrcoef(features)
    distance = np.nan_to_num(distance, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = float(distance[np.triu_indices(len(distance), 1)].mean())
    if sigma <= 0:
        return np.ones_like(distance)
    return np.exp(-(distance**2) / (2.0 * sigma**2))


def graph_hash(edge_index: np.ndarray, edge_weight: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(edge_index, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(edge_weight, dtype=np.float64).tobytes())
    return digest.hexdigest()


class ParisotGCNAdapter:
    """Canonical graph builder; model training remains locked behind the formal gate."""

    upstream_commit = "185e2f5d9745484b8c7994f80254740b9cbb7b75"
    license = "GPL-3.0-or-later; clean-room adapter only"
    chebyshev_order = 3
    hidden_layers = 1
    hidden_channels = 16
    dropout = 0.3

    def __init__(self, fold: BaselineFold):
        self.fold = fold

    @classmethod
    def config(cls, **overrides: object) -> CanonicalModelConfig:
        values = dict(name="Parisot-GCN-C", layers=1, chebyshev_order=3)
        values.update(overrides)
        return CanonicalModelConfig(**values)

    @classmethod
    def model(cls, **overrides: object) -> CanonicalPopulationModel:
        return CanonicalPopulationModel(cls.config(**overrides))

    def transductive_graph(self) -> tuple[np.ndarray, np.ndarray]:
        phenotype = (
            (self.fold.sex[:, None] == self.fold.sex[None, :]).astype(float)
            + (self.fold.site[:, None] == self.fold.site[None, :]).astype(float)
        )
        weight = phenotype * _correlation_similarity(self.fold.features)
        np.fill_diagonal(weight, 0.0)
        source, target = np.nonzero(weight)
        return np.vstack((source, target)).astype(np.int64), weight[source, target]

    def strict_query_graph(
        self, query_indices: tuple[int, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        references = np.asarray(self.fold.boundary.optimization_train, dtype=int)
        queries = np.asarray(query_indices, dtype=int)
        if set(references) & set(queries):
            raise ValueError("query overlaps optimization training")
        edges, weights = [], []
        for query_position, query in enumerate(queries):
            isolated = np.concatenate((references, np.asarray([query])))
            similarity = _correlation_similarity(self.fold.features[isolated])
            local_query = len(references) + query_position
            for local_reference, reference in enumerate(references):
                phenotype = int(self.fold.sex[reference] == self.fold.sex[query])
                phenotype += int(self.fold.site[reference] == self.fold.site[query])
                value = phenotype * similarity[local_reference, len(references)]
                if value > 0:
                    edges.append((local_reference, local_query))
                    weights.append(value)
        return np.asarray(edges, dtype=np.int64).T, np.asarray(weights, dtype=np.float64)


@dataclass
class TrainOnlyFeatureStandardizer:
    """Explicit fit-boundary object shared by future baseline runners."""

    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    fit_indices: tuple[int, ...] | None = None

    def fit(self, values: np.ndarray, optimization_train: tuple[int, ...]):
        if not optimization_train or len(set(optimization_train)) != len(optimization_train):
            raise ValueError("invalid optimization-training boundary")
        selected = np.asarray(values, dtype=np.float64)[list(optimization_train)]
        self.mean = selected.mean(0)
        self.scale = selected.std(0)
        self.scale[self.scale == 0] = 1.0
        self.fit_indices = tuple(optimization_train)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("standardizer is not fitted")
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.scale
