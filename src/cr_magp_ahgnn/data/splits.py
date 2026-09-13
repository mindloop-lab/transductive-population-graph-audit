"""PA-001 split-manifest validation and explicit partition boundaries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cr_magp_ahgnn.config.schema import EXPECTED_SPLIT_SHA256
from .registry import sha256_file


@dataclass(frozen=True)
class FoldBoundary:
    fold: int
    optimization_train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]


@dataclass(frozen=True)
class SplitManifest:
    path: Path
    sha256: str
    subject_count: int
    folds: tuple[FoldBoundary, ...]

    @classmethod
    def load(cls, path: Path, expected_sha256: str = EXPECTED_SPLIT_SHA256) -> "SplitManifest":
        actual = sha256_file(path)
        if actual != expected_sha256:
            raise ValueError(f"split SHA-256 mismatch: expected {expected_sha256}, got {actual}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        count = int(payload["subject_count"])
        folds = tuple(
            FoldBoundary(
                fold=int(row["fold"]),
                optimization_train=tuple(row["optimization_train"]["indices"]),
                validation=tuple(row["validation"]["indices"]),
                test=tuple(row["test"]["indices"]),
            )
            for row in payload["folds"]
        )
        instance = cls(path.resolve(), actual, count, folds)
        instance.validate()
        return instance

    def validate(self) -> None:
        if self.subject_count != 871 or len(self.folds) != 10:
            raise ValueError("split must cover 871 subjects in 10 folds")
        universe = set(range(self.subject_count))
        coverage = [0] * self.subject_count
        for boundary in self.folds:
            parts = [set(boundary.optimization_train), set(boundary.validation), set(boundary.test)]
            if any(parts[i] & parts[j] for i in range(3) for j in range(i + 1, 3)):
                raise ValueError(f"partition overlap in fold {boundary.fold}")
            if set.union(*parts) != universe:
                raise ValueError(f"incomplete partition in fold {boundary.fold}")
            for index in boundary.test:
                coverage[index] += 1
        if any(value != 1 for value in coverage):
            raise ValueError("test partitions must provide exact-once OOF coverage")
