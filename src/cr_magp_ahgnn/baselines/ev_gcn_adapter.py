"""Clean-room EV-GCN canonical adapter.

Reference: SamitHuang/EV_GCN@57363b85c56ee9d093976bda203ef62c237c985f
(GPL-3.0). No GPL source is copied or imported.
"""

from __future__ import annotations

import numpy as np

from .parisot_gcn_adapter import BaselineFold, ParisotGCNAdapter
from .canonical_model import CanonicalModelConfig, CanonicalPopulationModel


class EVGCNAdapter:
    upstream_commit = "57363b85c56ee9d093976bda203ef62c237c985f"
    license = "GPL-3.0; clean-room adapter only"
    chebyshev_order = 3
    graph_layers = 4
    hidden_channels = 16
    edge_dropout = 0.3

    def __init__(self, fold: BaselineFold):
        self.fold = fold

    @classmethod
    def config(cls, **overrides: object) -> CanonicalModelConfig:
        values = dict(
            name="EV-GCN-C", layers=4, chebyshev_order=3, learned_edges=True,
            layer_concat=True, learning_rate=0.01, weight_decay=5e-5,
        )
        values.update(overrides)
        return CanonicalModelConfig(**values)

    @classmethod
    def model(cls, **overrides: object) -> CanonicalPopulationModel:
        return CanonicalPopulationModel(cls.config(**overrides))

    def candidate_graph(self) -> tuple[np.ndarray, np.ndarray]:
        edges, _ = ParisotGCNAdapter(self.fold).transductive_graph()
        phenotype = np.column_stack((self.fold.sex, self.fold.site)).astype(np.float64)
        source, target = edges
        edge_features = np.concatenate((phenotype[source], phenotype[target]), axis=1)
        train = np.asarray(self.fold.boundary.optimization_train, dtype=int)
        train_mask = np.isin(source, train) & np.isin(target, train)
        fit_values = edge_features[train_mask]
        if not len(fit_values):
            raise ValueError("no optimization-training edges")
        mean, scale = fit_values.mean(0), fit_values.std(0)
        scale[scale == 0] = 1.0
        return edges, (edge_features - mean) / scale

    def strict_query_candidates(self, query_indices: tuple[int, ...]) -> np.ndarray:
        edges, _ = ParisotGCNAdapter(self.fold).strict_query_graph(query_indices)
        return edges
