#!/usr/bin/env python3
"""Build STATISTICAL_LINEAGE_MAP.csv for Paper M from its own phase-2 statistics package.

Sources (Paper M's own archive, not Paper A's PA_* files):
  statistics/paired_effects.csv            raw two-sided subject-bootstrap p per contrast
  _stage/canonical_result_registry.csv     prediction file + sha256 + split hashes + identity status
  docs/statistical_methods.md              resampling rules and family statement
Missing items are recorded as missing; nothing is inferred from displayed p-values and nothing is
borrowed from another paper.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

S = Path("/mnt/disk_8tb/codex/2609-wan/paperM/PAPERM_ARCHIVE_261003/03_results/phase2_statistics")
R3 = Path("/mnt/disk_8tb/codex/2609-wan/paperM/PAPERM_CMIG_R3_20261011")

pe = list(csv.DictReader((S / "statistics/paired_effects.csv").open(encoding="utf-8", errors="ignore")))
reg = list(csv.DictReader((S / "_stage/canonical_result_registry.csv").open(encoding="utf-8", errors="ignore")))
by_id = {r["experiment_id"]: r for r in reg}
# the phase-2 identity mapping (different naming convention: P1a_head_*, P1b_*, p1c_*)
_idmap_p = S / "_stage/experiment_identity_mapping.csv"
idmap = list(csv.DictReader(_idmap_p.open(encoding="utf-8", errors="ignore"))) if _idmap_p.exists() else []
# the phase-2 paired table and the identity mapping use different naming conventions; resolve by
# normalised alias matching (case-insensitive, order-insensitive tokens) instead of declaring a
# contrast unresolved on a naming difference alone.
import re as _re


def _norm(s: str) -> str:
    return _re.sub(r"[^a-z0-9]+", "", s.lower())


_norm_index = {}
for r in reg:
    _norm_index.setdefault(_norm(r["experiment_id"]), r)
    _norm_index.setdefault(_norm(r["experiment_id"].split("_seed")[0]), r)
    base = r["experiment_id"].replace("P1a_head_", "p1a_").replace("P1b_", "p1b_").replace("p1c_", "p1c_")
    _norm_index.setdefault(_norm(base), r)


def resolve(cid: str):
    if cid in by_id:
        return by_id[cid], "exact"
    # 1. direct match on the prediction-file basename (the release ships <condition>_oof.npz)
    rel_root = Path("/mnt/disk_8tb/codex/2609-wan/paperM/PAPERM_CMIG_R3_20261011/released_repo")
    for sub in ("results/e1_out/p1_out", "results/e1_out/transductive_controls",
                "results/e1_out/loso_cpac", "results/e1_out/loso_niak"):
        for f in sorted((rel_root / sub).glob("*_oof.npz")) if (rel_root / sub).exists() else []:
            if f.name == f"{cid}_oof.npz":
                return {"experiment_id": cid, "prediction_file": f"{sub}/{f.name}",
                        "prediction_sha256": "", "generation_split_hash": "",
                        "identity_status": "PREDICTION_FILE_PRESENT_IN_RELEASE",
                        "protocol": "", "in_canonical_family": ""}, "oof-basename"
    # 2. normalised / token-subset alias matching against both registry and identity mapping
    n = _norm(cid)
    if n in _norm_index:
        return _norm_index[n], "normalised"
    toks = set(_re.findall(r"[a-z]+", cid.lower())) - {"seed666", "seed2025", "seed2026", "s666"}
    best, hits = None, 0
    for cand in list(reg) + list(idmap):
        ct = set(_re.findall(r"[a-z]+", cand["experiment_id"].lower()))
        score = len(toks & ct)
        if toks and score == len(toks) and score > hits:
            best, hits = cand, score
    if best is not None:
        return best, "token-subset"
    return {}, "unresolved"

rows, unresolved = [], []
for i, r in enumerate(pe, start=1):
    ids = [x.strip() for x in r["contrast"].split(" vs ")]
    a, mode_a = resolve(ids[0])
    b, mode_b = resolve(ids[1])
    both_known = bool(a) and bool(b)
    same_protocol = bool(a) and bool(b) and a.get("protocol") == b.get("protocol")
    rel = Path("/mnt/disk_8tb/codex/2609-wan/paperM/PAPERM_CMIG_R3_20261011/released_repo")
    rows.append({
        "row": i,
        "contrast_id": r["contrast"],
        "condition_a": ids[0], "condition_b": ids[1],
        "auc_a": r["auc_a"], "auc_b": r["auc_b"], "delta_auc": r["delta_auc"],
        "delta_ci_low": r["delta_ci_low"], "delta_ci_high": r["delta_ci_high"],
        "protocol": r["protocol"], "resampling_unit": r["resampling_unit"],
        "replicates": r["replicates"], "invalid_resamples": r["invalid"],
        "paired_identity_same_subject_set": r["paired_identity_same_subject_set"],
        "raw_p_source": "statistics/paired_effects.csv::p_two_sided_subject_boot",
        "raw_p": r["p_two_sided_subject_boot"],
        "raw_p_note": ("resolution-limited: bootstrap with invalid single-class resamples dropped "
                       "and counted; 0.0 means no valid resample reached the extreme"),
        "oof_a_path": a.get("prediction_file", ""), "oof_a_sha256": a.get("prediction_sha256", ""),
        "oof_b_path": b.get("prediction_file", ""), "oof_b_sha256": b.get("prediction_sha256", ""),
        "generation_split_hash_a": a.get("generation_split_hash", ""),
        "identity_status_a": a.get("identity_status", ""), "identity_status_b": b.get("identity_status", ""),
        "in_canonical_family_a": a.get("in_canonical_family", ""),
        "statistics_script": "phase2_statistics/scripts/paired_identity_tests.py; "
                             "phase2_statistics/_stage/fold_hypothesis_tests.py",
        "family": r["protocol"],
        "inferential_status": ("within-protocol paired inference"
                               if same_protocol else "CROSS-PROTOCOL: non-inferential"),
        "both_conditions_found_in_registry": both_known,
        "condition_a_resolution": mode_a, "condition_b_resolution": mode_b,
        "oof_a_in_release": bool(a.get("prediction_file")) and (rel / a["prediction_file"]).exists(),
        "oof_b_in_release": bool(b.get("prediction_file")) and (rel / b["prediction_file"]).exists(),
    })
    if not both_known:
        unresolved.append({"contrast_id": r["contrast"],
                           "missing": [x for x in ids if x not in by_id]})

fields = list(rows[0].keys()) if rows else ["row"]
with (R3 / "STATISTICAL_LINEAGE_MAP.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow(r)

fam = {}
for r in rows:
    fam.setdefault(r["family"], []).append(r["contrast_id"])
coverage = {
    "rows_mapped": len(rows),
    "families": {k: len(v) for k, v in fam.items()},
    "contrasts_with_both_conditions_in_registry": sum(1 for r in rows if r["both_conditions_found_in_registry"]),
    "unresolved_contrasts": unresolved,
    "manuscript_declares": {"contrasts_total": 13, "within_protocol": 12},
    "coverage_verdict": ("PARTIAL: the phase-2 package yields "
                         f"{len(rows)} contrast rows with raw p and OOF identity; the manuscript's stated "
                         "13-item family (12 within-protocol + 1 cross-protocol) is not fully covered by this "
                         "file, so the remaining items still need to be located before any recomputation."),
    "discipline": ["no borrowing from Paper A files", "no back-derivation from displayed p-values",
                   "missing items recorded as missing"],
    "other_paper_candidates_excluded": [
        "results/canonical/PA_SOTA002_PAIRED_COMPARISONS_CANONICAL.csv (PA_ prefix -> Paper A)",
        "results/canonical/PA_HOLM_CORRECTION_CANONICAL.csv (PA_ prefix -> Paper A)",
        "results/canonical/PA_PROTOCOL_EFFECTS_CANONICAL.csv (PA_ prefix -> Paper A)",
    ],
}
(R3 / "STATISTICAL_LINEAGE_COVERAGE.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2),
                                                      encoding="utf-8")
print("mapped rows:", coverage["rows_mapped"], "| families:", coverage["families"])
print("both conditions in registry:", coverage["contrasts_with_both_conditions_in_registry"])
print("unresolved:", coverage["unresolved_contrasts"])
st = json.loads((R3 / "PLAN_EXECUTION_STATE.json").read_text(encoding="utf-8"))
st["open_dependencies"]["holms_raw_p"] = (
    "PARTIALLY RECOVERED from Paper M's own phase-2 package: raw two-sided subject-bootstrap p, "
    "per-contrast conditions, OOF paths+sha256 and protocol families mapped "
    f"({coverage['rows_mapped']} rows). The stated 13-item family (12 within-protocol + 1 "
    "cross-protocol) is not fully covered yet -> see STATISTICAL_LINEAGE_COVERAGE.json. "
    "Recomputation stays blocked until coverage and method validity are complete.")
(R3 / "PLAN_EXECUTION_STATE.json").write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
print("state updated")