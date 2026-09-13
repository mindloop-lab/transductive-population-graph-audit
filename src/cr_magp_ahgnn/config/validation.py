"""Configuration and output-boundary validation."""

from __future__ import annotations

from pathlib import Path

from .schema import EXPECTED_SPLIT_SHA256, Phase0AConfig


def validate_config(config: Phase0AConfig, canonical_roots: list[Path] | None = None) -> None:
    if config.split_sha256 != EXPECTED_SPLIT_SHA256:
        raise ValueError("configured split SHA-256 is not PA-001-authorized")
    if config.n_folds != 10 or config.seed != 666:
        raise ValueError("Phase 0A requires 10 folds and recovered seed 666")
    if not config.canonical_read_only or not config.external_data_locked:
        raise ValueError("canonical read-only and external-data lock must remain enabled")
    output = config.run_root.resolve()
    for root in canonical_roots or []:
        canonical = root.resolve()
        if output == canonical or canonical in output.parents:
            raise ValueError(f"run root must not be inside canonical data: {canonical}")
