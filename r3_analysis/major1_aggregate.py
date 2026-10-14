#!/usr/bin/env python3
"""Major 1 formal aggregation (protocol-locked quantities only).

Primary   : per-site macro of (AUC_ON - AUC_OFF) over the site's feasible folds, computed per
            training seed and then averaged over the three fixed seeds.
Secondary : pooled OOF AUC over the site's test subjects (states whether it is subject-weighted).
CI        : subject-level bootstrap inside each site (test-fold subjects only; the paired ON/OFF
            scores are resampled together), B=10,000, seed 20261012, percentile interval.
Integrity : every fold's AUC is recomputed from the saved OOF logits and compared with the value in
            the run report; a mismatch aborts. Fold AUCs with a degenerate class are marked, kept in
            the record and excluded from the primary mean, as specified in the protocol.

Final mode refuses to emit numbers unless all 528 fits are present. --dry-run emits marked
partial output for code validation only.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from _common import load_reports, load_predictions, validate_pairs

R3 = Path("reproduced/r3")
RUNS = R3 / "major1_runs"
SEEDS = [666, 2025, 2026]
B = 10_000
BOOT_SEED = 20261012


def auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return None
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (pos.size * neg.size)


def auc_matrix(s: np.ndarray, labels: np.ndarray):
    """Return (gt, eq, npos, nneg) matrices for the folded-cross-product bootstrap."""
    n = s.size
    gt = (s[:, None] > s[None, :]).astype(np.float64)
    eq = (s[:, None] == s[None, :]).astype(np.float64)
    pos = (labels == 1).astype(np.float64)
    neg = (labels == 0).astype(np.float64)
    return gt, eq, pos, neg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--boot", type=int, default=B)
    ap.add_argument("--runs", default=str(RUNS), help="directory with the per-fit reports and OOF arrays")
    ap.add_argument("--out-csv", default=str(R3 / "major1_site_level.csv"))
    ap.add_argument("--out-json", default=str(R3 / "major1_bootstrap.json"))
    args = ap.parse_args()
    B_local = args.boot

    reps, expected = load_reports(Path(args.manifest), RUNS_DIR := Path(args.runs),
                                  allow_partial=args.dry_run)
    by = {(r["cell"], r["arm"], r["seed"]): r for r in reps}
    cells_meta = defaultdict(set)
    for r in reps:
        cells_meta[r["site"]].add(r["cell"])
    if not reps:
        raise SystemExit("refusing: no official runs")
    validate_pairs(reps, RUNS_DIR)

    # --- integrity: recompute each fold AUC from OOF logits and compare with the report ---
    mismatches, degenerate = [], 0
    fold_scores: dict = {}
    for r in reps:
        z = load_predictions(RUNS_DIR, r)
        role = np.asarray(z["oof_role"])
        lab = np.asarray(z["oof_label"])
        lg = np.asarray(z["oof_logits"], dtype=np.float64)
        ex = np.exp(lg - lg.max(axis=1, keepdims=True))
        sc = (ex / ex.sum(axis=1, keepdims=True))[:, 1]  # softmax positive-class probability, as the harness scores it
        m = role == "T"
        a = auc(sc[m], lab[m])
        if a is None:
            degenerate += 1
        elif abs(a - r["test_auc"]) > 1e-9:
            mismatches.append((r["tag"], a, r["test_auc"]))
        fold_scores[(r["cell"], r["arm"], r["seed"])] = dict(
            s=sc[m], lab=lab[m], subj=np.asarray(z["subject_ids"])[m], fold=r["fold"], site=r["site"],
            auc=a, report_auc=r["test_auc"])
    if mismatches:
        raise SystemExit(f"ABORT: {len(mismatches)} recomputed AUCs disagree with reports, e.g. {mismatches[:3]}")

    rng = np.random.default_rng(BOOT_SEED)
    site_rows, boot_records = [], {}
    for site, cs in sorted(cells_meta.items()):
        cs = sorted(cs)
        usable = [c for c in cs if all(k in fold_scores for k in
                                       [(c, a, s) for a in ("ON", "OFF") for s in SEEDS])]
        if not usable:
            continue
        per_seed, per_seed_folds = {}, {}
        for s in SEEDS:
            ds, used = [], 0
            for c in usable:
                on, off = fold_scores[(c, "ON", s)], fold_scores[(c, "OFF", s)]
                if on["auc"] is None or off["auc"] is None:
                    continue
                ds.append(on["auc"] - off["auc"])
                used += 1
            if used:
                per_seed[s] = float(np.mean(ds))
                per_seed_folds[s] = used
        if not per_seed:
            continue
        seed_vals = np.array([per_seed[s] for s in sorted(per_seed)])
        primary = float(seed_vals.mean())

        # --- subject-level bootstrap inside the site, paired ON/OFF, folds fixed ---
        # For each seed and fold build the pairwise matrices once; a replicate only reweights subjects.
        mats = {}
        for s in SEEDS:
            for c in usable:
                on, off = fold_scores[(c, "ON", s)], fold_scores[(c, "OFF", s)]
                mats[(s, c)] = (auc_matrix(on["s"], on["lab"]), auc_matrix(off["s"], off["lab"]))
        site_subj = sorted({t for c in usable for s in SEEDS
                            for t in fold_scores[(c, "ON", s)]["subj"]})
        idx = {t: i for i, t in enumerate(site_subj)}
        nsub = len(site_subj)
        repl = np.empty(B_local)
        for b in range(B_local):
            w = np.bincount(rng.integers(0, nsub, nsub), minlength=nsub).astype(np.float64)
            seed_d = []
            for s in sorted(per_seed):
                ds = []
                for c in usable:
                    (gt_o, eq_o, p_o, n_o), (gt_f, eq_f, p_f, n_f) = mats[(s, c)]
                    wi = np.array([w[idx[t]] for t in fold_scores[(c, "ON", s)]["subj"]])
                    wp = np.array([w[idx[t]] for t in fold_scores[(c, "OFF", s)]["subj"]])
                    Wo = np.outer(wi, wi)
                    np_, nn_ = float((wi * p_o).sum()), float((wi * n_o).sum())
                    if np_ <= 0 or nn_ <= 0:
                        continue
                    ao = float((Wo * (gt_o + 0.5 * eq_o) * np.outer(p_o, n_o)).sum()) / (np_ * nn_)
                    Wf = np.outer(wp, wp)
                    npf, nnf = float((wp * p_f).sum()), float((wp * n_f).sum())
                    if npf <= 0 or nnf <= 0:
                        continue
                    af = float((Wf * (gt_f + 0.5 * eq_f) * np.outer(p_f, n_f)).sum()) / (npf * nnf)
                    ds.append(ao - af)
                if ds:
                    seed_d.append(float(np.mean(ds)))
            repl[b] = float(np.mean(seed_d)) if seed_d else np.nan
        lo, hi = np.nanpercentile(repl, [2.5, 97.5])
        # internal consistency gate: the bootstrap must reproduce its own point estimate
        gate_gap = abs(float(np.nanmean(repl)) - primary)
        if gate_gap > 0.05:
            raise SystemExit(f"ABORT: bootstrap for {site} does not reproduce the point estimate "
                             f"(repl mean {np.nanmean(repl):+.3f} vs point {primary:+.3f}); "
                             f"the CI is not reportable until this is fixed")
        boot_records[site] = {"primary_mean": primary, "ci95": [float(lo), float(hi)],
                              "b": B_local, "seed": BOOT_SEED,
                              "n_site_test_subjects": nsub}
        # secondary: pooled OOF AUC over the site's test subjects (subject-weighted)
        pools = {}
        for s in SEEDS:
            on = np.concatenate([fold_scores[(c, "ON", s)]["s"] for c in usable])
            lo_ = np.concatenate([fold_scores[(c, "ON", s)]["lab"] for c in usable])
            off = np.concatenate([fold_scores[(c, "OFF", s)]["s"] for c in usable])
            lo2 = np.concatenate([fold_scores[(c, "OFF", s)]["lab"] for c in usable])
            pools[s] = (auc(on, lo_), auc(off, lo2))
        site_rows.append({"site": site, "folds_used": len(usable), "primary": primary,
                          "per_seed": per_seed, "folds_per_seed": per_seed_folds,
                          "ci_lo": float(lo), "ci_hi": float(hi),
                          "pooled_on": float(np.mean([pools[s][0] for s in pools])),
                          "pooled_off": float(np.mean([pools[s][1] for s in pools]))})

    tag = "DRY-RUN PARTIAL — NOT A RESULT" if args.dry_run else "FINAL"
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_csv).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["site", "folds_used", "primary", "ci_lo", "ci_hi",
                                          "pooled_on", "pooled_off",
                                          "per_seed", "folds_per_seed"])
        w.writeheader()
        for r in site_rows:
            w.writerow({**r, "per_seed": json.dumps(r["per_seed"]), "folds_per_seed": json.dumps(r["folds_per_seed"])})
    out = {"status": tag, "fits_on_disk": len(reps), "sites_reported": len(site_rows),
           "recomputed_auc_mismatches": 0, "degenerate_fold_aucs": degenerate,
           "primary_definition": "site macro of (AUC_ON-AUC_OFF), then mean over seeds {666,2025,2026}",
           "bootstrap": {"b": B_local, "seed": BOOT_SEED, "unit": "test-fold subjects within site, paired",
                         "sites": boot_records,
                         "note": "no pairwise sign pattern here may be promised as a positive effect"},
           "inference_status": "conditional intervals only; withdrawn per-site tests and Holm values are not computed"}
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{tag}: {len(site_rows)} sites | fits {len(reps)} | degenerate {degenerate} | "
          f"mismatches 0 | -> {Path(args.out_csv).name}, {Path(args.out_json).name}")
    for r in site_rows[:10]:
        print(f"  {r['site']:<10} folds {r['folds_used']:<2} primary {r['primary']:+.3f} "
              f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]")


if __name__ == "__main__":
    main()