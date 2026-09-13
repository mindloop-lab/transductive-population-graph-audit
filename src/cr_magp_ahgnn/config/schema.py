"""Side-effect-free Phase 0A configuration schema."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


EXPECTED_SPLIT_SHA256 = "45539f252a822f5267face1b52d7c3bb2c5fed6b90020442f3340aaff7d7cce4"


@dataclass(frozen=True)
class Phase0AConfig:
    data_registry: Path
    split_manifest: Path
    run_root: Path
    seed: int = 666
    n_folds: int = 10
    pca_components: int = 32
    split_sha256: str = EXPECTED_SPLIT_SHA256
    canonical_read_only: bool = True
    external_data_locked: bool = True

    @classmethod
    def from_json(cls, path: Path) -> "Phase0AConfig":
        values = json.loads(path.read_text(encoding="utf-8"))
        base = path.resolve().parent
        for key in ("data_registry", "split_manifest", "run_root"):
            value = Path(values[key])
            values[key] = value if value.is_absolute() else base / value
        return cls(**values)

    @classmethod
    def from_environment(cls) -> "Phase0AConfig":
        required = ("CR_MAGP_DATA_REGISTRY", "CR_MAGP_SPLIT_MANIFEST", "CR_MAGP_RUN_ROOT")
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            raise ValueError(f"missing required environment variables: {', '.join(missing)}")
        return cls(
            data_registry=Path(os.environ[required[0]]),
            split_manifest=Path(os.environ[required[1]]),
            run_root=Path(os.environ[required[2]]),
        )

    def to_dict(self) -> dict:
        values = asdict(self)
        for key in ("data_registry", "split_manifest", "run_root"):
            values[key] = str(values[key])
        return values
