#!/usr/bin/env python3
"""Validate the frozen public result registry and its release provenance.

Dependency-free on purpose, so it can run in CI before the scientific stack is installed.
It checks the registry schema, the value ranges, the seed metadata, the backing artifacts
and -- importantly -- the split provenance: the per-seed generation partition hashes and
the aggregation reference are re-derived here from the shipped artifacts rather than
trusted.

Run:  python scripts/validate_registry.py [registry.csv]
Exit: 0 clean, 1 on any violation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
REG = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "result_registry_paperM_v4.csv"
GEN_PARTITIONS = ROOT / "data_contract" / "oof_generation_outer_folds.json"
AGG_REFERENCE = ROOT / "data_contract" / "phase_minus1b_splits.json"

# Assembled from fragments so this file itself carries no literal machine path; the
# public-tree hygiene scan can then demand a strict zero over the tracked tree.
_MACHINE_PREFIXES = ("/" + "mnt" + "/", "/" + "Users" + "/", "/" + "home" + "/",
                     "file" + ":///")

EXPECTED = [
    "experiment_id", "model", "pipeline", "protocol", "graph_setting",
    "mechanism_intervention", "seed", "metric", "score_definition", "estimate",
    "ci_low", "ci_high", "ci_subj_low", "ci_subj_high", "n", "n_positive",
    "prediction_file", "generation_split_hash", "aggregation_split_hash",
    "code_commit", "note",
]
NUMERIC_01 = ("estimate", "ci_low", "ci_high", "ci_subj_low", "ci_subj_high")
TEXT_COLS = (
    "experiment_id", "model", "pipeline", "protocol", "graph_setting",
    "mechanism_intervention", "metric", "score_definition", "prediction_file",
    "generation_split_hash", "aggregation_split_hash", "code_commit", "note",
)
GEN_STATUSES = ("not_redistributed", "not_recovered_outer_partition",
                "not_applicable_site_wise_loso")


def is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def pairs_payload(pairs) -> bytes:
    def key(kv):
        k = kv[0]
        return (0, int(k)) if str(k).lstrip("-").isdigit() else (1, str(k))
    return ("\n".join(f"{k}\t{v}" for k, v in sorted(pairs, key=key)) + "\n").encode("utf-8")


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# ---- re-derive the provenance objects from the shipped artifacts -----------------
known_partitions: dict[str, str] = {}
partition_errors: list[str] = []
if GEN_PARTITIONS.is_file():
    gen = json.loads(GEN_PARTITIONS.read_text())
    for seed, block in gen.get("partitions", {}).items():
        pairs = [(sid, fold) for fold, ids in block["folds"].items() for sid in ids]
        recomputed = sha(pairs_payload(pairs))
        if recomputed != block["hash"]:
            partition_errors.append(f"generation partition {seed}: recorded hash "
                                    f"{block['hash'][:16]}... != recomputed {recomputed[:16]}...")
        known_partitions[seed] = block["hash"]
else:
    partition_errors.append("data_contract/oof_generation_outer_folds.json is missing")

agg_hash = ""
if AGG_REFERENCE.is_file():
    man = json.loads(AGG_REFERENCE.read_text())
    pairs = [(str(sid), f["fold"]) for f in man["folds"] for sid in f["test"]["subject_ids"]]
    agg_hash = sha(pairs_payload(pairs))
else:
    partition_errors.append("data_contract/phase_minus1b_splits.json is missing")

with REG.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.reader(handle))

if not rows:
    raise SystemExit("FAIL: empty registry")

header, data = rows[0], rows[1:]
errors: list[str] = list(partition_errors)

if header != EXPECTED:
    errors.append(f"header mismatch: expected {EXPECTED!r}, got {header!r}")

for lineno, row in enumerate(data, start=2):
    if len(row) != len(header):
        errors.append(f"line {lineno}: {len(row)} fields, header has {len(header)}")
        continue

    d = dict(zip(header, row))
    rid = d.get("experiment_id") or f"line {lineno}"

    for col in NUMERIC_01:
        value = d.get(col, "")
        if not value:
            continue
        if not is_float(value):
            errors.append(f"{rid}: {col}={value!r} is not numeric")
        elif not 0.0 <= float(value) <= 1.0:
            errors.append(f"{rid}: {col}={value} outside [0,1]")

    est = d.get("estimate", "")
    lo = d.get("ci_low", "")
    hi = d.get("ci_high", "")
    if lo and est and is_float(lo) and is_float(est) and float(lo) > float(est):
        errors.append(f"{rid}: ci_low > estimate")
    if hi and est and is_float(hi) and is_float(est) and float(hi) < float(est):
        errors.append(f"{rid}: ci_high < estimate")

    for col in ("n", "n_positive"):
        value = d.get(col, "")
        if value and not value.isdigit():
            errors.append(f"{rid}: {col}={value!r} is not an integer")
    if d.get("n", "").isdigit() and d.get("n_positive", "").isdigit():
        if int(d["n_positive"]) > int(d["n"]):
            errors.append(f"{rid}: n_positive > n")

    for col in TEXT_COLS:
        value = d.get(col, "")
        if value and is_float(value):
            errors.append(f"{rid}: {col} parses as a float ({value!r}) -- column shift?")

    if not d.get("metric"):
        errors.append(f"{rid}: empty metric")

    pred = d.get("prediction_file", "")
    if pred:
        path = pathlib.Path(pred)
        if path.is_absolute() or ".." in path.parts:
            errors.append(f"{rid}: prediction_file must be repository-relative")
        elif not (ROOT / path).is_file():
            errors.append(f"{rid}: backing artifact not found: {pred}")
    elif "not redistributed" not in d.get("note", "").lower():
        errors.append(f"{rid}: empty prediction_file without an explicit not-redistributed note")

    seed = d.get("seed", "")
    if seed.isdigit():
        match = re.search(r"(?:_s|seed)(\d{3,4})(?:_|$)", rid, flags=re.IGNORECASE)
        if match and match.group(1) != seed:
            errors.append(f"{rid}: id encodes seed {match.group(1)} but seed column is {seed}")

    # split provenance: a partition hash of the generation run, an explicit status, or
    # (for aggregated rows only) nothing in this column.
    gensplit = d.get("generation_split_hash", "")
    if not gensplit:
        errors.append(f"{rid}: empty generation_split_hash")
    elif gensplit in GEN_STATUSES:
        pass
    elif gensplit not in known_partitions.values():
        errors.append(f"{rid}: generation_split_hash {gensplit[:16]}... is not a shipped "
                      f"generation partition")
    else:
        # the recorded partition must be the one that could have produced this row
        match = re.search(r"(?:seed|_s)(\d{3,4})", pathlib.Path(pred).name)
        if match and match.group(1) in known_partitions:
            if known_partitions[match.group(1)] != gensplit:
                errors.append(f"{rid}: generation_split_hash does not match the partition of "
                              f"seed {match.group(1)}")

    aggsplit = d.get("aggregation_split_hash", "-")
    if aggsplit not in ("-", agg_hash):
        errors.append(f"{rid}: aggregation_split_hash is neither '-' nor the shipped "
                      f"aggregation reference")
    if d.get("metric") == "mean_fold_AUC" and aggsplit != agg_hash:
        errors.append(f"{rid}: mean-fold row must cite the aggregation reference")

    if not d.get("code_commit"):
        errors.append(f"{rid}: empty code_commit/provenance")

    if any(prefix in value for value in row for prefix in _MACHINE_PREFIXES):
        errors.append(f"{rid}: private filesystem path leaked into registry")

print(f"registry rows checked: {len(data)}; generation partitions verified: "
      f"{len(known_partitions)}; aggregation reference: {'ok' if agg_hash else 'missing'}")
if errors:
    print(f"FAIL: {len(errors)} release-integrity violation(s)")
    for error in errors[:100]:
        print("  -", error)
    raise SystemExit(1)

print("PASS: registry schema, ranges, backing artifacts, split provenance and seed "
      "metadata are valid")
