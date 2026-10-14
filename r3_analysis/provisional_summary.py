#!/usr/bin/env python3
"""Provisional Major-1 summary (campaign still running).

Computes, for every COMPLETE site (all feasible folds with both arms and all three seeds), the
protocol's primary quantity: per seed, the site-macro of (AUC_ON - AUC_OFF) over that site's
feasible folds; then the arithmetic mean over the three fixed seeds. Pooled OOF AUC over the site's
test subjects is reported as the secondary quantity. Values are PROVISIONAL while the campaign runs
and are not interpreted here.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

R3 = Path("/mnt/disk_8tb/codex/2609-wan/paperM/PAPERM_CMIG_R3_20261011")
RUNS = R3 / "major1_runs"
SEEDS = [666, 2025, 2026]

reps = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RUNS.glob("*_report.json"))]
by = {(r["cell"], r["arm"], r["seed"]): r for r in reps}
cells = sorted({r["cell"] for r in reps})
sites = {}
for r in reps:
    sites.setdefault(r["site"], set()).add(r["cell"])

rows = []
for site, cs in sorted(sites.items()):
    complete_cells = [c for c in sorted(cs)
                      if all((c, a, s) in by for a in ("ON", "OFF") for s in SEEDS)]
    if not complete_cells:
        continue
    per_seed = {}
    for s in SEEDS:
        diffs = [by[(c, "ON", s)]["test_auc"] - by[(c, "OFF", s)]["test_auc"] for c in complete_cells]
        per_seed[s] = sum(diffs) / len(diffs)
    macro = sum(per_seed.values()) / len(per_seed)
    rows.append({"site": site, "n_cells_complete": len(complete_cells),
                 "folds_total_feasible": len(cs),
                 "d_s666": round(per_seed[666], 4), "d_s2025": round(per_seed[2025], 4),
                 "d_s2026": round(per_seed[2026], 4),
                 "site_macro_mean_over_seeds": round(macro, 4),
                 "stability_note": "PROVISIONAL; site may still be missing folds",
                 "fits_on_disk": len(reps)})
if rows:
    out = R3 / "MAJOR1_PROVISIONAL_SUMMARY.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"PROVISIONAL (fits on disk {len(reps)}/528) — complete sites: {len(rows)}")
    for r in rows:
        print(f"  {r['site']:<10} folds {r['n_cells_complete']}/{r['folds_total_feasible']}  "
              f"d666 {r['d_s666']:+.3f}  d2025 {r['d_s2025']:+.3f}  d2026 {r['d_s2026']:+.3f}  "
              f"mean {r['site_macro_mean_over_seeds']:+.3f}")
    print(f"-> {out}")
else:
    print(f"no site complete yet (fits on disk {len(reps)}/528)")
# unified ledger refresh
uni = R3 / "MAJOR1_RUN_LEDGER_UNIFIED.csv"
with uni.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["tag", "cell", "site", "fold", "arm", "seed", "test_auc",
                                      "best_val", "best_epoch", "updates", "epochs_run",
                                      "supervised_n", "init_sha12", "mask_sha12"])
    w.writeheader()
    for r in reps:
        w.writerow({"tag": r["tag"], "cell": r["cell"], "site": r["site"], "fold": r["fold"],
                    "arm": r["arm"], "seed": r["seed"], "test_auc": round(r["test_auc"], 4),
                    "best_val": round(r["best_val_auc"], 4), "best_epoch": r["best_epoch"],
                    "updates": r["updates"], "epochs_run": r["epochs_run"],
                    "supervised_n": r["supervised"]["n"], "init_sha12": r["init_sha256"][:12],
                    "mask_sha12": r["mask_sha256"][:12]})
print(f"unified ledger rows -> {len(reps)}")