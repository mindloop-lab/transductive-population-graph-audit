"""Subject-level, exact-once OOF accumulator."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


class OOFAccumulator:
    def __init__(self, subject_ids: tuple[str, ...], labels: np.ndarray):
        if len(subject_ids) != len(labels) or len(set(subject_ids)) != len(subject_ids):
            raise ValueError("OOF registry and labels are inconsistent")
        self.subject_ids = subject_ids
        self.labels = np.asarray(labels, dtype=np.int64)
        self.logits = np.full((len(labels), 2), np.nan, dtype=np.float64)
        self.fold = np.full(len(labels), -1, dtype=np.int64)

    def add_fold(self, fold: int, indices: tuple[int, ...], logits: np.ndarray) -> None:
        indices_array = np.asarray(indices, dtype=np.int64)
        values = np.asarray(logits, dtype=np.float64)
        if values.shape != (len(indices), 2):
            raise ValueError("fold logits must have shape [test_subjects, 2]")
        if np.any(self.fold[indices_array] >= 0):
            raise ValueError("OOF subject assigned more than once")
        self.logits[indices_array] = values
        self.fold[indices_array] = fold

    def validate_complete(self) -> None:
        if np.isnan(self.logits).any() or np.any(self.fold < 0):
            raise ValueError("OOF coverage is incomplete")

    def save_csv(self, path: Path) -> None:
        self.validate_complete()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["subject_id", "fold", "label", "logit_0", "logit_1"])
            for index, subject_id in enumerate(self.subject_ids):
                writer.writerow([subject_id, int(self.fold[index]), int(self.labels[index]), *self.logits[index]])
