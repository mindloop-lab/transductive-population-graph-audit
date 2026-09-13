"""Independent strict-query adapter for the single-subject PopGNN protocol.

Reference: 98jaemin/single_subject_popgnn@3522c83cf2ac19a8488baeb444dadf57a62fc35a.
The upstream snapshot has no tracked license; no source is copied or imported.
"""

from __future__ import annotations

import numpy as np

from .parisot_gcn_adapter import BaselineFold, _correlation_similarity
from .canonical_model import CanonicalModelConfig, CanonicalPopulationModel


class SingleSubjectPopGNNAdapter:
    upstream_commit = "3522c83cf2ac19a8488baeb444dadf57a62fc35a"
    license = "NO TRACKED LICENSE; independent protocol adapter only"
    graph_layers = 4
    hidden_channels = 16
    default_test_batch = 1

    def __init__(self, fold: BaselineFold, *, threshold: float = 1.1):
        self.fold = fold
        self.threshold = float(threshold)

    @classmethod
    def config(cls, **overrides: object) -> CanonicalModelConfig:
        values = dict(
            name="SingleSubject-PopGNN-C", layers=4, chebyshev_order=1,
            learned_edges=True, residual=True, layer_concat=True,
            learning_rate=0.001, weight_decay=5e-5,
        )
        values.update(overrides)
        return CanonicalModelConfig(**values)

    @classmethod
    def model(cls, **overrides: object) -> CanonicalPopulationModel:
        return CanonicalPopulationModel(cls.config(**overrides))

    def strict_query_edges(self, query_indices: tuple[int, ...]) -> np.ndarray:
        references = np.asarray(self.fold.boundary.optimization_train, dtype=int)
        queries = np.asarray(query_indices, dtype=int)
        if set(references) & set(queries):
            raise ValueError("query overlaps optimization training")
        edges = []
        for query_position, query in enumerate(queries):
            isolated = np.concatenate((references, np.asarray([query])))
            similarity = _correlation_similarity(self.fold.features[isolated])
            q_local = len(references) + query_position
            for r_local, reference in enumerate(references):
                phenotype = int(self.fold.sex[reference] == self.fold.sex[query])
                phenotype += int(self.fold.site[reference] == self.fold.site[query])
                if phenotype * similarity[r_local, len(references)] > self.threshold:
                    edges.append((r_local, q_local))
        return np.asarray(edges, dtype=np.int64).reshape(-1, 2).T
