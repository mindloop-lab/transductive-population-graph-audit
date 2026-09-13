"""Read-only view of the frozen ABIDE-I registry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DataRegistry:
    manifest_path: Path
    payload: dict

    @classmethod
    def load(cls, path: Path, *, verify_files: bool = True) -> "DataRegistry":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("dataset") != "ABIDE-I" or payload.get("cohort", {}).get("subjects") != 871:
            raise ValueError("registry is not the PA-001 ABIDE-I 871-subject cohort")
        registry = cls(path.resolve(), payload)
        if verify_files:
            registry.verify_authority_files()
        return registry

    @property
    def subject_ids_path(self) -> Path:
        return Path(self.payload["paths"]["subject_ids"])

    @property
    def canonical_subject_root(self) -> Path:
        return Path(self.payload["paths"]["subject_root"])

    def subject_ids(self) -> tuple[str, ...]:
        values = tuple(line.strip() for line in self.subject_ids_path.read_text().splitlines() if line.strip())
        if len(values) != 871 or len(set(values)) != 871:
            raise ValueError("subject registry must contain 871 unique IDs")
        return values

    def verify_authority_files(self) -> None:
        expected = self.payload["sha256"]
        checks = {
            "subject_ids": self.subject_ids_path,
            "id_mapping": Path(self.payload["paths"]["id_mapping"]),
            "phenotype": Path(self.payload["paths"]["phenotype"]),
            "matrix_hash_manifest": Path(self.payload["paths"]["matrix_hash_manifest"]),
        }
        for name, path in checks.items():
            if not path.is_file() or sha256_file(path) != expected[name]:
                raise ValueError(f"registry authority check failed: {name}")
