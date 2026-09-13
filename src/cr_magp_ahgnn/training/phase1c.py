"""Locked Phase 1C development-pilot contracts and small training helpers."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from cr_magp_ahgnn.config.schema import EXPECTED_SPLIT_SHA256
from cr_magp_ahgnn.cr import CRClassifier
from cr_magp_ahgnn.models import M0Strict


AUTHORIZED_FOLDS = (0, 1, 2)
AUTHORIZED_VARIANTS = ("C0", "C1", "C2")


@dataclass(frozen=True)
class Phase1CConfig:
    split_manifest: Path
    split_sha256: str
    raw_fixture: Path
    phase0d_root: Path
    seed: int
    folds: tuple[int, ...]
    variants: tuple[str, ...]
    protocol: str
    k: int
    temperature: float
    alpha_init: float
    gate_bias_init: float
    base_model_frozen: bool
    reference_labels_as_input: bool
    optimizer: str
    learning_rate: float
    weight_decay: float
    max_epochs: int
    patience: int
    minimum_delta: float
    selection_metric: str
    threshold_source: str
    test_evaluations_per_fold: int
    external_data_locked: bool
    formal_sota_claim_allowed: bool
    phase: str = "1C"
    scope: str = "development_pilot"

    @classmethod
    def load(cls, path: Path) -> "Phase1CConfig":
        values = json.loads(path.read_text(encoding="utf-8"))
        base = path.resolve().parent
        for key in ("split_manifest", "raw_fixture", "phase0d_root"):
            value = Path(values[key])
            values[key] = value if value.is_absolute() else (base / value).resolve()
        values["folds"] = tuple(values["folds"])
        values["variants"] = tuple(values["variants"])
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        exact = {
            "seed": 666,
            "folds": AUTHORIZED_FOLDS,
            "variants": AUTHORIZED_VARIANTS,
            "protocol": "strict_single_query",
            "k": 8,
            "temperature": 0.5,
            "alpha_init": 0.0,
            "gate_bias_init": -2.0,
            "base_model_frozen": True,
            "reference_labels_as_input": False,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0005,
            "max_epochs": 40,
            "patience": 8,
            "minimum_delta": 1e-6,
            "selection_metric": "validation_auc_plus_balanced_accuracy",
            "threshold_source": "validation_only",
            "test_evaluations_per_fold": 1,
            "external_data_locked": True,
            "formal_sota_claim_allowed": False,
            "phase": "1C",
            "scope": "development_pilot",
        }
        for name, expected in exact.items():
            if getattr(self, name) != expected:
                raise ValueError(f"PA-005 locked configuration mismatch: {name}")
        if self.split_sha256 != EXPECTED_SPLIT_SHA256:
            raise ValueError("PA-005 split SHA mismatch")
        if not self.raw_fixture.is_file() or not self.phase0d_root.is_dir():
            raise ValueError("frozen Phase 0D inputs are unavailable")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reserve_output(path: Path) -> None:
    resolved = path.resolve()
    forbidden = {"canonical", "evidence", "MAGP_AHGNN-v2"}
    if any(part in forbidden for part in resolved.parts):
        raise ValueError("refusing canonical/evidence output path")
    resolved.mkdir(parents=True, exist_ok=False)


def save_exclusive(path: Path, payload: dict) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        torch.save(payload, stream)


def initialize_cr(m0: M0Strict, variant: str, gate_bias_init: float = -2.0) -> CRClassifier:
    if variant not in AUTHORIZED_VARIANTS:
        raise ValueError("unknown Phase 1C variant")
    for parameter in m0.parameters():
        parameter.requires_grad_(False)
    m0.eval()
    model = CRClassifier(m0, m0.subject_encoder.output_dim, 32)
    final_gate = model.gate.network[-1]
    with torch.no_grad():
        final_gate.bias.fill_(gate_bias_init)
    if variant in ("C0", "C1"):
        for parameter in model.gate.parameters():
            parameter.requires_grad_(False)
    if variant == "C0":
        for parameter in model.correction.parameters():
            parameter.requires_grad_(False)
    return model


def optimizer_parameters(model: CRClassifier, variant: str) -> list[torch.nn.Parameter]:
    allowed_prefixes = ("correction.",) if variant == "C1" else ("gate.", "correction.")
    if variant == "C0":
        return []
    values = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith(allowed_prefixes)
    ]
    if not values or any(parameter.requires_grad for parameter in model.m0.parameters()):
        raise AssertionError("optimizer/base freeze contract violated")
    return values


def parameter_manifest(model: CRClassifier, variant: str) -> dict:
    optimizer_ids = {id(value) for value in optimizer_parameters(model, variant)}
    rows = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "numel": parameter.numel(),
            "requires_grad": parameter.requires_grad,
            "optimizer_member": id(parameter) in optimizer_ids,
        }
        for name, parameter in model.named_parameters()
    ]
    return {
        "variant": variant,
        "base_model_frozen": not any(parameter.requires_grad for parameter in model.m0.parameters()),
        "optimizer_parameter_count": sum(row["numel"] for row in rows if row["optimizer_member"]),
        "parameters": rows,
    }


def topk_affinities(distances: torch.Tensor, *, k: int, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    if distances.ndim != 2 or distances.shape[1] < k or temperature <= 0:
        raise ValueError("invalid top-k affinity input")
    values, indices = torch.topk(distances, k, dim=1, largest=False, sorted=True)
    weights = torch.softmax(-values / temperature, dim=1)
    return indices, weights


def validation_threshold(labels: torch.Tensor, probabilities: torch.Tensor) -> float:
    y = labels.detach().cpu().numpy()
    p = probabilities.detach().cpu().numpy()
    candidates = np.unique(np.concatenate(([0.5], p)))
    ranked = []
    for threshold in candidates:
        score = balanced_accuracy_score(y, (p >= threshold).astype(np.int64))
        ranked.append((float(score), -abs(float(threshold) - 0.5), -float(threshold), float(threshold)))
    return max(ranked)[-1]


def metrics(labels: torch.Tensor, probabilities: torch.Tensor, threshold: float) -> dict[str, float]:
    y = labels.detach().cpu().numpy()
    p = probabilities.detach().cpu().numpy()
    prediction = (p >= threshold).astype(np.int64)
    tp = int(((prediction == 1) & (y == 1)).sum())
    tn = int(((prediction == 0) & (y == 0)).sum())
    positive = int((y == 1).sum())
    negative = int((y == 0).sum())
    confidence = np.maximum(p, 1.0 - p)
    correct = (prediction == y).astype(np.float64)
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (confidence >= lower) & (confidence < upper if lower < 0.9 else confidence <= 1.0)
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    return {
        "auc": float(roc_auc_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "sensitivity": tp / positive,
        "specificity": tn / negative,
        "f1": float(f1_score(y, prediction)),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": ece,
        "positive_rate": float(prediction.mean()),
        "threshold": float(threshold),
    }


def two_step_gradient_audit() -> dict:
    seed_everything(666)
    m0 = M0Strict(5, 7, hidden=4)
    model = initialize_cr(m0, "C2")
    query = torch.tensor([[0.3, -0.2, 0.1, 0.7, -0.4, 0.8, -0.6, 0.5]])
    references = torch.tensor(
        [[[0.2, 0.4, -0.3, 0.1, 0.8, -0.5, 0.6, -0.7], [0.9, -0.8, 0.7, -0.6, 0.5, -0.4, 0.3, -0.2]]]
    )
    references.requires_grad_(False)
    labels = torch.tensor([1])
    first = model(query, references, torch.tensor([[0.4, 0.6]]))
    first_loss = functional.cross_entropy(first.logits, labels)
    first_loss.backward()
    alpha_grad = model.correction.alpha.grad
    first_alpha_ok = alpha_grad is not None and torch.isfinite(alpha_grad) and alpha_grad.abs() > 0
    if not bool(first_alpha_ok):
        raise AssertionError("alpha gradient gate failed at zero initialization")
    with torch.no_grad():
        model.correction.alpha.add_(-0.01 * alpha_grad)
    model.zero_grad(set_to_none=True)
    affinity_proxy = torch.tensor([[0.4, 0.6]], requires_grad=True)
    second = model(query, references, affinity_proxy)
    second.reference_context.retain_grad()
    functional.cross_entropy(second.logits, labels).backward()
    gate_grads = [parameter.grad for parameter in model.gate.parameters()]
    residual_grads = [model.correction.alpha.grad, model.correction.projection.weight.grad]

    def finite_nonzero(values) -> bool:
        return all(value is not None and torch.isfinite(value).all() and value.abs().sum() > 0 for value in values)

    result = {
        "alpha_initial": 0.0,
        "first_alpha_grad": float(alpha_grad.detach()),
        "first_alpha_grad_finite_nonzero": bool(first_alpha_ok),
        "second_aggregator_output_grad_finite_nonzero": finite_nonzero([second.reference_context.grad]),
        "second_affinity_proxy_grad_finite_nonzero": finite_nonzero([affinity_proxy.grad]),
        "second_gate_grad_finite_nonzero": finite_nonzero(gate_grads),
        "second_residual_grad_finite_nonzero": finite_nonzero(residual_grads),
        "frozen_m0_grad_count": sum(parameter.grad is not None for parameter in model.m0.parameters()),
        "frozen_reference_requires_grad": references.requires_grad,
        "frozen_reference_grad_is_none": references.grad is None,
    }
    if not all(
        result[name]
        for name in (
            "second_aggregator_output_grad_finite_nonzero",
            "second_affinity_proxy_grad_finite_nonzero",
            "second_gate_grad_finite_nonzero",
            "second_residual_grad_finite_nonzero",
            "frozen_reference_grad_is_none",
        )
    ) or result["frozen_m0_grad_count"] != 0:
        raise AssertionError("Phase 1C second-step/frozen gradient gate failed")
    return result
