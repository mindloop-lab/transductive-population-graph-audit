#!/usr/bin/env python3
"""Compare the symmetrised rerun with the canonical Major 1 run, using the SAME inference procedure.

For each version: per-site macro of (AUC_ON - AUC_OFF) per training seed, averaged over the three seeds;
then the population macro (equal-weight mean over sites), a site-resample bootstrap interval (B=10,000,
seed 20261012) and an exact two-sided sign test -- identical to major1_population_inference.py.

The symmetrised version is a robustness parallel analysis (author decision (b)+(c)); it never replaces
the primary numbers.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

PAPERM = Path("/mnt/disk_8tb/codex/2609-wan/paperM")
R3 = PAPERM / "PAPERM_CMIG_R3_20261011"
SEEDS = [666, 2025, 2026]
B, BOOT_SEED = 10_000, 20261012


MANIFEST = json.loads((R3 / "SPLIT_AND_MASK_MANIFEST.json").read_text(encoding="utf-8"))
ALLOWED = {f"cell{i:03d}_{c['site']}_f{c['fold']}_{arm}_s{s}"
           for i, c in enumerate(MANIFEST["cells"]) if c["feasible"]
           for arm in ("ON", "OFF") for s in MANIFEST["seeds"]["training"]}


def site_effects(runs: Path):
    """Select analysis units BY MANIFEST NAME, so smoke or stray reports in the directory cannot enter."""
    files = [p for p in sorted(runs.glob("*_report.json")) if p.name[:-len("_report.json")] in ALLOWED]
    skipped = [p.name for p in sorted(runs.glob("*_report.json")) if p.name[:-len("_report.json")] not in ALLOWED]
    if skipped:
        print(f"  [unit filter] skipped {len(skipped)} non-manifest reports in {runs.name}: {skipped[:3]}")
    reps = [json.loads(p.read_text(encoding="utf-8")) for p in files]
    by = {(r["cell"], r["arm"], r["seed"]): r for r in reps}
    cells = defaultdict(set)
    for r in reps:
        cells[r["site"]].add(r["cell"])
    eff, foldcount = {}, {}
    for site, cs in sorted(cells.items()):
        cs = sorted(cs)
        per_seed = {}
        for s in SEEDS:
            ds = [by[(c, "ON", s)]["test_auc"] - by[(c, "OFF", s)]["test_auc"] for c in cs
                  if (c, "ON", s) in by and (c, "OFF", s) in by]
            if ds:
                per_seed[s] = float(np.mean(ds))
        if per_seed:
            eff[site] = float(np.mean(list(per_seed.values())))
            foldcount[site] = len(cs)
    return eff, foldcount, len(reps)


def infer(eff: dict) -> dict:
    sites = sorted(eff)
    v = np.array([eff[s] for s in sites])
    n = v.size
    pop = float(v.mean())
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n, size=(B, n))
    reps = v[idx].mean(axis=1)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    # two-sided null-centred p: tails defined by the ABSOLUTE observed effect (valid for either sign),
    # with the standard finite-simulation correction (b+1)/(B+1) instead of a bare floor
    centred = v - pop
    c = centred[idx].mean(axis=1)
    beyond = int((np.abs(c) >= abs(pop)).sum())
    p = min(1.0, (beyond + 1) / (B + 1))
    # exact two-sided sign test: use the SMALLER tail (valid when the count is below n/2)
    k = int((v > 0).sum())
    kk = min(k, n - k)
    p_sign = min(1.0, 2 * sum(comb(n, i) for i in range(kk + 1)) / 2 ** n)
    return {"n_sites": n, "population_macro": pop, "ci95": [float(lo), float(hi)],
            "null_centred_p": float(p), "sign_k": k, "sign_p": float(p_sign),
            "per_site": {s: eff[s] for s in sites}}


ap = argparse.ArgumentParser()
ap.add_argument("--runs-a", default=str(R3 / "major1_runs"))
ap.add_argument("--runs-b", default=str(R3 / "major1_runs_sym"))
ap.add_argument("--out-json", default=str(R3 / "SYMMETRISED_VS_CANONICAL.json"))
ap.add_argument("--out-csv", default=str(R3 / "SYMMETRISED_VS_CANONICAL.csv"))
ap.add_argument("--allow-partial", action="store_true",
                help="write the JSON even when the symmetrised run is incomplete (never used by the final chain)")
args = ap.parse_args()

can, can_folds, can_n = site_effects(Path(args.runs_a))
sym, sym_folds, sym_n = site_effects(Path(args.runs_b))
if not args.allow_partial and (can_n != 528 or sym_n != 528):
    raise SystemExit(f"refusing: canonical {can_n}/528, symmetrised {sym_n}/528 fits; "
                     "use --allow-partial for a non-recordable check")
out = {"complete": bool(can_n == 528 and sym_n == 528),
       "status": "robustness parallel analysis (author decision (b)+(c)); primary numbers unchanged",
       "canonical": {**infer(can), "folds": can_folds, "fits": can_n},
       "symmetrised": {**infer(sym), "folds": sym_folds, "fits": sym_n}}
common = sorted(set(can) & set(sym))          # only sites present in BOTH versions are compared
out["per_site_side_by_side"] = {s: {"canonical": can[s], "symmetrised": sym[s],
                                   "same_sign": (can[s] > 0) == (sym[s] > 0)} for s in common}
out["sign_agreement"] = {"agree": sum(1 for s in common if out["per_site_side_by_side"][s]["same_sign"]),
                         "comparable_sites": len(common),
                         "canonical_only": sorted(set(can) - set(sym)),
                         "symmetrised_only": sorted(set(sym) - set(can))}
(Path(args.out_json)).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

with Path(args.out_csv).open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["site", "canonical_site_macro", "symmetrised_site_macro", "same_sign"])
    for s in common:
        w.writerow([s, can[s], sym[s], out["per_site_side_by_side"][s]["same_sign"]])
print("canonical   :", can_n, "fits | pop %+0.4f CI [%+0.4f,%+0.4f] sign %d/%d p=%.3g"
      % (out["canonical"]["population_macro"], *out["canonical"]["ci95"], out["canonical"]["sign_k"],
         out["canonical"]["n_sites"], out["canonical"]["sign_p"]))
print("symmetrised :", sym_n, "fits | pop %+0.4f CI [%+0.4f,%+0.4f] sign %d/%d p=%.3g"
      % (out["symmetrised"]["population_macro"], *out["symmetrised"]["ci95"], out["symmetrised"]["sign_k"],
         out["symmetrised"]["n_sites"], out["symmetrised"]["sign_p"]))
print("per-site sign agreement (comparable sites only):", out["sign_agreement"])