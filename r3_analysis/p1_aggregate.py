#!/usr/bin/env python3
"""P1 aggregation: support-sampling sensitivity. The ON arm is NOT a determinism control (that
hypothesis was tested and refuted), and the graph does NOT change across draws: the ON arm's supervised
target set is C+S with C=O\\R, so its label composition changes with the draw.

For each site and replicate: the site macro of (AUC_ON - AUC_OFF) over that site's folds, per training
seed, then averaged over seeds. Then across the three replicates: the mean and the spread.

Control: the ON arm does not depend on the support draw, so its results must be identical across
replicates for the same cell and seed. Any disagreement is reported as a cross-replicate inconsistency
smoothed over.

Decision rule (frozen before the run): if the across-replicate spread of the site effect is comparable
to the effect itself, supervision location and support composition are not separable at this sample
size and the manuscript must say so; if the spread is small, the entanglement limitation can be
narrowed in wording, but causal wording is still not licensed.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

R3 = Path("/mnt/disk_8tb/codex/2609-wan/paperM/PAPERM_CMIG_R3_20261011")
RUNS = R3 / "P1_support_runs"
SEEDS = [666, 2025, 2026]
SITES = ["LEUVEN_1", "PITT", "YALE"]

reps = {}
for p in sorted(RUNS.glob("r*_report.json")):
    r = json.loads(p.read_text(encoding="utf-8"))
    tag = p.name[:-len("_report.json")]
    r["_replicate"] = int(tag.split("_", 1)[0][1:])
    reps[(r["_replicate"], r["cell"], r["arm"], r["seed"])] = r

# cross-replicate comparison of the ON arm (invariance hypothesis REFUTED; graph is fixed)
on_control, failures = 0, []   # differing ON pairs across replicates
by_cell_seed = defaultdict(dict)
for (rep, cell, arm, seed), r in reps.items():
    if arm == "ON":
        by_cell_seed[(cell, seed)][rep] = r["test_auc"]
for (cell, seed), d in by_cell_seed.items():
    if len(d) > 1:
        on_control += 1                     # every pair with >=2 replicates is comparable
        vals = set(round(v, 12) for v in d.values())
        if len(vals) > 1:
            failures.append((cell, seed, d))

per_site = {}
for site in SITES:
    cell_folds = sorted({c for (_rep, c, _a, _s), r in reps.items() if r["site"] == site})
    per_rep = {}
    for rep in sorted({k[0] for k in reps}):
        per_seed = {}
        for s in SEEDS:
            ds = []
            for c in cell_folds:
                a = reps.get((rep, c, "ON", s)); b = reps.get((rep, c, "OFF", s))
                if a and b:
                    ds.append(a["test_auc"] - b["test_auc"])
            if ds:
                per_seed[s] = sum(ds) / len(ds)
        if per_seed:
            per_rep[rep] = {"per_seed": per_seed, "mean_over_seeds": sum(per_seed.values()) / len(per_seed),
                            "folds_used": len(cell_folds)}
    if per_rep:
        vals = [v["mean_over_seeds"] for v in per_rep.values()]
        per_site[site] = {"per_replicate": per_rep,
                          "across_replicate_mean": sum(vals) / len(vals),
                          "across_replicate_spread": max(vals) - min(vals),
                          "complete_replicates": len(per_rep)}

out = {
    "status": "sensitivity analysis (frozen design); not a primary result",
    "sites": per_site,
    "on_arm_cross_replicate": {"pairs_compared": on_control, "differing_pairs": len(failures),
                               "hypothesis": "ON arm invariant across draws: REFUTED",
                               "corrected_mechanism": "graph fixed across draws; supervised target set C+S with C=O\\R changes with the draw",
                               "examples": failures[:5]},
    "frozen_decision_rule": "spread comparable to the effect => location and composition not separable",
}
(R3 / "P1_SUPPORT_SENSITIVITY.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

with (R3 / "P1_SUPPORT_SENSITIVITY.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["site", "replicate", "mean_over_seeds", "spread_across_replicates", "folds_used"])
    for site, d in per_site.items():
        for rep, v in d["per_replicate"].items():
            w.writerow([site, rep, round(v["mean_over_seeds"], 4), round(d["across_replicate_spread"], 4), v["folds_used"]])

print(f"fits on disk: {len(reps)} | ON pairs compared across replicates: {on_control} | differing: {len(failures)}")
for site, d in per_site.items():
    vals = " ".join(f"r{r}:{v['mean_over_seeds']:+.3f}" for r, v in d["per_replicate"].items())
    print(f"  {site:<9} {vals}  | mean {d['across_replicate_mean']:+.3f} | spread {d['across_replicate_spread']:.3f}")
if failures:
    print("  ON arm differs across replicates in", len(failures), "pairs (invariance hypothesis refuted; caused by the changing supervised target set, not the graph)")