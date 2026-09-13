"""Fold-local scaler/PCA with serialized fit-boundary evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class FoldPreprocessor:
    n_components: int = 32
    scaler: StandardScaler | None = None
    pca: PCA | None = None
    fit_indices: tuple[int, ...] | None = None

    def fit(self, subject_features: list[np.ndarray], optimization_train: tuple[int, ...]) -> "FoldPreprocessor":
        if not optimization_train or len(set(optimization_train)) != len(optimization_train):
            raise ValueError("optimization-training indices must be non-empty and unique")
        stacked = np.vstack([subject_features[index] for index in optimization_train])
        self.scaler = StandardScaler().fit(stacked)
        normalized = self.scaler.transform(stacked)
        self.pca = PCA(n_components=self.n_components).fit(normalized) if normalized.shape[1] > self.n_components else None
        self.fit_indices = tuple(int(index) for index in optimization_train)
        return self

    def transform(self, subject_features: list[np.ndarray]) -> list[np.ndarray]:
        if self.scaler is None or self.fit_indices is None:
            raise RuntimeError("preprocessor has not been fit")
        output = []
        for features in subject_features:
            normalized = self.scaler.transform(features)
            output.append(self.pca.transform(normalized) if self.pca is not None else normalized)
        return output

    def save(self, path: Path, *, fold: int, split_sha256: str) -> Path:
        if self.scaler is None or self.fit_indices is None:
            raise RuntimeError("cannot save an unfitted preprocessor")
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "fold": fold,
            "split_sha256": split_sha256,
            "fit_partition": "optimization_train_only",
            "fit_indices": list(self.fit_indices),
            "fit_indices_sha256": hashlib.sha256(json.dumps(self.fit_indices).encode()).hexdigest(),
        }
        joblib.dump({"metadata": metadata, "scaler": self.scaler, "pca": self.pca}, path)
        return path
