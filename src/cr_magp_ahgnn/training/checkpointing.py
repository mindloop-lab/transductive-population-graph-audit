"""Atomic run metadata and file-hash manifests; no model training logic."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch
import torch_geometric

from cr_magp_ahgnn.config.schema import Phase0AConfig


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.strip()


def write_run_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def hash_artifacts(paths: list[Path]) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in paths}


def create_run_manifest(config: Phase0AConfig, repository_root: Path, artifacts: list[Path]) -> dict:
    """Capture the minimum reproducibility envelope without starting a run."""
    return {
        "schema_version": 1,
        "phase": "0A",
        "training_performed": False,
        "cr_enabled": False,
        "git_commit": git_commit(repository_root),
        "seed": config.seed,
        "configuration": config.to_dict(),
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "artifact_sha256": hash_artifacts(artifacts),
    }
