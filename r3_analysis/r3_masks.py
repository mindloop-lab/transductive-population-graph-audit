#!/usr/bin/env python3
"""R3 Major 1 — build the paired supervision-location design: splits, masks, feasibility.

Design (frozen here, before any training):
  eligible site s: both classes have >= MIN_PER_CLASS labelled subjects
  target-site 5-fold stratified split -> per fold: T = test fold, S = remaining same-site subjects
  V = fixed validation subjects drawn only from OTHER sites (never target-site labels)
  O = other-site trainable subjects excluding V;  R = O sampled with per-class counts equal to S
  C = O \\ R  (shared supervised subjects);  arms: SAME_SITE_ON uses labels from C+S,
  SAME_SITE_OFF uses labels from C+R  -> identical supervised totals and class counts
Feasibility depends only on counts/splits/input completeness, never on performance.
Writes SPLIT_AND_MASK_MANIFEST.json, MAJOR1_FEASIBILITY.csv, protocol.yaml, and updates the state.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

PAPERM = Path("/mnt/disk_8tb/codex/2609-wan/paperM")
R3 = PAPERM / "PAPERM_CMIG_R3_20261011"
CACHE = Path("/mnt/disk_8tb/codex/2609-wan/del/MAGP_AHGNN-v2/artifacts/phase_minus1b/cache_abide_data.npz")
PHENO = Path("/mnt/disk_8tb/data/abide_pcp/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv")

SEED_SPLIT = 20261011          # splitter seed (frozen)
SEED_SUPPORT = 20261012        # support-sampling seed, independent of training seeds
MIN_PER_CLASS = 5
MIN_TEST_PER_CLASS = 2   # amendment 2026-10-11: a test fold must carry >=2 of each class, defined
                         # before any formal run and depending only on the split, never on results
N_FOLDS = 5
TRAIN_SEEDS = [666, 2025, 2026]
VAL_FRACTION = 0.10
VAL_MAX = 120

ids = [str(x) for x in np.load(CACHE)["subject_ids"]]
N = len(ids)
pheno = {}
with PHENO.open(newline="", encoding="utf-8", errors="ignore") as f:
    for row in csv.DictReader(f):
        pheno[str(int(row["SUB_ID"]))] = row
y = np.array([1 if int(pheno[str(int(s))]["DX_GROUP"]) == 1 else 0 for s in ids])
site = np.array([pheno[str(int(s))]["SITE_ID"] for s in ids])
sites = sorted(set(site.tolist()))

rng_split = np.random.default_rng(SEED_SPLIT)
rng_support = np.random.default_rng(SEED_SUPPORT)

# ---------------------------------------------------------------- fixed validation set (other sites only)
all_idx = np.arange(N)

cells, feas_rows = [], []
for s in sites:
    s_idx = np.where(site == s)[0]
    n_asd, n_td = int((y[s_idx] == 1).sum()), int((y[s_idx] == 0).sum())
    eligible = n_asd >= MIN_PER_CLASS and n_td >= MIN_PER_CLASS
    if not eligible:
        feas_rows.append({"site": s, "fold": "", "eligible": False,
                          "reason": f"class counts {n_asd}/{n_td} below {MIN_PER_CLASS}"})
        continue
    # fixed per-site validation set: drawn only from OTHER sites, identical across folds and arms
    rng_val = np.random.default_rng(SEED_SPLIT + 1000 + sites.index(s))
    other_sites_idx = np.setdiff1d(all_idx, s_idx)
    V = np.sort(rng_val.choice(other_sites_idx,
                               size=min(VAL_MAX, int(round(len(other_sites_idx) * VAL_FRACTION))),
                               replace=False))
    # stratified 5-fold within the site (test labels used only by the splitter, then sealed)
    a = s_idx[y[s_idx] == 1].copy(); t = s_idx[y[s_idx] == 0].copy()
    rng_split.shuffle(a); rng_split.shuffle(t)
    for k in range(N_FOLDS):
        T = np.sort(np.concatenate([a[k::N_FOLDS], t[k::N_FOLDS]]))
        S = np.sort(np.setdiff1d(s_idx, T))
        O = np.setdiff1d(all_idx, np.concatenate([s_idx, V]))
        ok = (len(np.intersect1d(T, V)) == 0 and len(np.intersect1d(S, V)) == 0
              and len(np.intersect1d(T, O)) == 0)
        need = Counter(y[S].tolist())
        r_asd = int((y[O] == 1).sum()); r_td = int((y[O] == 0).sum())
        matched = r_asd >= need[1] and r_td >= need[0]
        R = np.array([], dtype=int)
        if matched:
            oa = O[y[O] == 1].copy(); ot = O[y[O] == 0].copy()
            rng_support.shuffle(oa); rng_support.shuffle(ot)
            R = np.sort(np.concatenate([oa[:need[1]], ot[:need[0]]]))
        C = np.sort(np.setdiff1d(O, R))
        budget_ok = (len(R) == len(S)) and Counter(y[R].tolist()) == need if matched else False
        _tc = Counter(y[T].tolist())
        test_ok = (_tc.get(1, 0) >= MIN_TEST_PER_CLASS) and (_tc.get(0, 0) >= MIN_TEST_PER_CLASS)
        # eligibility is site-level (>= MIN_PER_CLASS per class); the fold criterion is that each
        # test fold must be NON-EMPTY in both classes (the site-level rule already guarantees it)
        fold_min = min(Counter(y[T].tolist()).get(0, 0), Counter(y[T].tolist()).get(1, 0))
        fold_ok = fold_min >= 1
        cell = {
            "site": s, "fold": k,
            "T_test": T.tolist(), "S_same_site_support": S.tolist(), "V_validation": V.tolist(),
            "O_other_trainable": O.tolist(), "R_out_of_site_support": R.tolist(), "C_common": C.tolist(),
            "n": {"T": len(T), "S": len(S), "V": len(V), "O": len(O), "R": len(R), "C": len(C)},
            "class_counts": {"S": dict(need), "R": dict(Counter(y[R].tolist())) if matched else {},
                             "T": dict(Counter(y[T].tolist()))},
            "feasible": bool(matched and budget_ok and ok and test_ok),
            "invariants": {"T_S_V_O_pairwise_disjoint": ok, "V_has_no_target_site": bool(ok),
                           "budget_and_class_match": bool(budget_ok),
                           "test_fold_min_per_class": bool(test_ok),
                           "test_fold_nonempty_in_both_classes": bool(fold_ok),
                           "criteria_note": ("primary criterion is the 2026-10-11 amendment "
                                             "MIN_TEST_PER_CLASS=2 per class per test fold; the "
                                             "instruction-literal reading (non-empty fold) is recorded "
                                             "here as the alternative and would admit 100 cells")},
            "arm_supervision": {"ON": {"n_supervised": int(len(C) + len(S)),
                                       "classes": dict(Counter(y[np.concatenate([C, S])].tolist()))},
                                "OFF": {"n_supervised": int(len(C) + len(R)) if matched else None,
                                        "classes": dict(Counter(y[np.concatenate([C, R])].tolist())) if matched else {}}},
        }
        ids_blob = "|".join(f"{k2}:" + ",".join(map(str, cell[k2])) for k2 in
                            ("T_test", "S_same_site_support", "V_validation", "O_other_trainable",
                             "R_out_of_site_support", "C_common"))
        cell["mask_sha256"] = hashlib.sha256(ids_blob.encode()).hexdigest()
        cells.append(cell)
        feas_rows.append({"site": s, "fold": k, "eligible": True, "reason": "",
                          "n_T": len(T), "n_S": len(S), "n_R": len(R), "n_C": len(C), "n_V": len(V),
                          "feasible": cell["feasible"], "mask_sha256": cell["mask_sha256"][:16]})

with (R3 / "MAJOR1_FEASIBILITY.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=sorted({k for r in feas_rows for k in r}))
    w.writeheader()
    for r in feas_rows:
        w.writerow(r)

feasible_cells = [c for c in cells if c["feasible"]]
manifest = {
    "design": "paired supervision-location experiment (Major 1)",
    "seeds": {"split": SEED_SPLIT, "support_sampling": SEED_SUPPORT, "training": TRAIN_SEEDS},
    "constants": {"n_folds": N_FOLDS, "min_per_class": MIN_PER_CLASS, "val_max": VAL_MAX,
                  "val_fraction": VAL_FRACTION, "pipeline": "C-PAC (primary)"},
    "n_sites": len(sites), "n_cells_total": len(cells), "n_cells_feasible": len(feasible_cells),
    "design_cap_fits": len(feasible_cells) * 2 * len(TRAIN_SEEDS),
    "cells": cells,
}
(R3 / "SPLIT_AND_MASK_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
(R3 / "protocol.yaml").write_text(
    "major1_supervision_location:\n"
    f"  split_seed: {SEED_SPLIT}\n  support_sampling_seed: {SEED_SUPPORT}\n"
    f"  training_seeds: {TRAIN_SEEDS}\n  n_folds: {N_FOLDS}\n"
    f"  min_per_class: {MIN_PER_CLASS}\n  validation: {{max_n: {VAL_MAX}, fraction: {VAL_FRACTION}, "
    "site_scope: other_sites_only}}\n"
    "  arms: [SAME_SITE_ON, SAME_SITE_OFF]\n"
    "  matched: {total_supervised: true, class_counts: true, graph_identical: true}\n"
    "  primary_metric: site_macro_paired_auc_difference_averaged_over_three_seeds\n"
    "  secondary_metric: pooled_oof_auc\n"
    "  primary_estimand: sensitivity_of_predictions_to_location_of_supervised_support\n"
    "  not_claimed: [literal_label_propagation, cross_architecture_generality, unseen_site_validity]\n",
    encoding="utf-8")

st = json.loads((R3 / "PLAN_EXECUTION_STATE.json").read_text(encoding="utf-8"))
st["open_dependencies"]["major1_masks"] = f"generated: {len(feasible_cells)} feasible cells"
st["open_dependencies"]["holms_raw_p"] = ("RECOVERABLE: full-precision raw p, permutation extreme counts, "
                                          "effects/CIs and family fields found in PA_SOTA002_PAIRED_"
                                          "COMPARISONS_CANONICAL.csv (+ PA_HOLM_CORRECTION_CANONICAL.csv)")
st["open_dependencies"]["seed_mapping"] = ("historical seed_scope is 666,2026,3407 -- the instruction's "
                                           "{666,2025,2026} must be mapped explicitly, not relabelled")
st["open_dependencies"]["naming_gap"] = ("baseline still contains 'does not denote a newly proposed model' "
                                         "(section 6.2 requires its removal)")
st["open_dependencies"]["parisot_impl_diagnosis"] = "not started"
st["entered"] = st["entered"]
(R3 / "PLAN_EXECUTION_STATE.json").write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"sites {len(sites)} | cells {len(cells)} | feasible {len(feasible_cells)} | design cap fits "
      f"{manifest['design_cap_fits']}")
print("per-site feasibility:", {r['site']: r.get('feasible', r.get('eligible')) for r in feas_rows
                                if r.get('fold', 0) == 0})
print("sizes example cell0:", {k: v for k, v in list(feasible_cells[0]["n"].items())} if feasible_cells else "none")
print("budget check example:", feasible_cells[0]["arm_supervision"] if feasible_cells else "none")