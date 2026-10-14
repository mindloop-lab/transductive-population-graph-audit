#!/usr/bin/env python3
"""Population-level inference for Major 1 — the pre-specified primary endpoint, implemented after
the fact and labelled as such.

Endpoint: equal-weight mean over the 18 sites of the within-site fold macro of (AUC_ON - AUC_OFF),
computed per training seed and then averaged over the three fixed seeds.

Reports, with the null hypothesis stated for each:
  * site-resample bootstrap CI: resample SITES with replacement (B=10,000, seed 20261012); statistics
    is the equal-weight site mean; interval is percentile.
  * null-centred bootstrap test: centre each site's contribution at the observed population mean
    before resampling, so the reference distribution is a no-effect distribution under exchangeable
    sites; two-sided p, reported with the 1/(B+1) floor (an exact zero is never reported).
  * exact sign test on the 18 site-level effects (two-sided, binomial).
  * Holm step-down across the 18 site-level null-centred p values.

This module does NOT replace the per-site conditional analysis in major1_site_level.csv; it consumes
it. It is a separate, documented analysis, not a pre-registered one.
"""
from __future__ import annotations

import csv
import json
from math import comb
from pathlib import Path

import numpy as np

R3 = Path("/mnt/disk_8tb/codex/2609-wan/paperM/PAPERM_CMIG_R3_20261011")
B = 10_000
SEED = 20261012
SEEDS = ["666", "2025", "2026"]

rows = list(csv.DictReader((R3 / "major1_site_level.csv").open(encoding="utf-8")))
sites = [r["site"] for r in rows]
per_seed = {r["site"]: json.loads(r["per_seed"]) for r in rows}
site_effect = np.array([np.mean([per_seed[r["site"]][s] for s in SEEDS if s in per_seed[r["site"]]])
                        for r in rows])           # site macro, mean over seeds (the primary unit)
seed_effects = {s: np.array([per_seed[r["site"]][s] for r in rows if s in per_seed[r["site"]]])
                for s in SEEDS}
n = len(sites)
pop = float(site_effect.mean())
per_seed_pop = {s: float(v.mean()) for s, v in seed_effects.items()}

rng = np.random.default_rng(SEED)
idx = rng.integers(0, n, size=(B, n))
# site-resample CI: equal-weight mean over resampled sites, seeds averaged inside the replicate
reps = np.array([seed_effects[s][idx[b]].mean() for b in range(B) for s in ["666"]] )
wide = np.zeros(B)
for b in range(B):
    wide[b] = np.mean([seed_effects[s][idx[b]].mean() for s in SEEDS])
lo, hi = np.percentile(wide, [2.5, 97.5])

# null-centred bootstrap: site contributions centred at their own mean -> no-effect reference
centred = site_effect - pop
cent_rep = np.array([centred[idx[b]].mean() for b in range(B)])
beyond = int((np.abs(cent_rep) >= abs(pop)).sum())
p_null = min(1.0, (beyond + 1) / (B + 1))

# exact sign test, two-sided
k_pos = int((site_effect > 0).sum())
kk = min(k_pos, n - k_pos)
p_sign = min(1.0, 2 * sum(comb(n, k) for k in range(kk + 1)) / 2 ** n)

# Per-site inference is deliberately NOT computed here. Testing a single site against the
# across-site distribution conflates two sampling units and would flag all 18 sites. A valid per-site
# test needs that site's own resampling unit (subjects within the site) with an explicitly stated
# null; until that is defined, per-site multiplicity correction is withheld rather than approximated.
site_p_note = ("withheld: no valid per-site raw p defined yet; multiplicity correction is not applied "
               "to per-site conditional intervals, which are reported as conditional quantities only")

out = {
    "status": "post-hoc analysis as required by external review 2026-10-13; NOT pre-registered",
    "endpoint": "equal-weight mean over the 18 sites of the within-site fold macro of (AUC_ON-AUC_OFF), "
                "per training seed then averaged over seeds {666,2025,2026}",
    "population_macro": pop,
    "population_macro_per_seed": per_seed_pop,
    "site_resample_ci95": [float(lo), float(hi)],
    "bootstrap": {"b": B, "seed": SEED, "unit": "sites resampled with replacement",
                  "null_centred_two_sided_p": float(p_null),
                  "p_floor": f"1/(B+1) = {1/(B+1):.3e}", "never_reports_exact_zero": True},
    "sign_test": {"k_positive": k_pos, "n": n, "two_sided_p": float(p_sign),
                  "null": "site-level effects are symmetric about zero"},
    "per_site": {r["site"]: {"site_macro_mean_over_seeds": float(site_effect[i]),
                             "raw_p": None, "holm": None}
                 for i, r in enumerate(rows)},
    "per_site_inference_note": site_p_note,
    "withdrawn": {"per_site_holm": "withheld (invalid unit)", "tail_ratio_p": "withdrawn"},
    "protocol_deviation": ("PROTOCOL_LOCK.md pre-specified a site-clustered bootstrap; the first "
                           "implementation resampled subjects within sites. This module implements the "
                           "pre-specified endpoint. The earlier per-site intervals remain valid only as "
                           "conditional-on-fixed-model-and-seed quantities and are labelled as such. The "
                           "earlier tail-ratio p values are withdrawn (not calibrated tests)."),
    "earlier_analysis": {"within_site_subject_bootstrap": "kept as conditional secondary analysis",
                         "tail_ratio_p": "withdrawn from any inference", "holm_on_tail_ratio_p": "withdrawn"},
}
(R3 / "major1_population_inference.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
print(f"population_macro {pop:+.4f}  per-seed " +
      " ".join(f"{s}:{per_seed_pop[s]:+.4f}" for s in SEEDS))
print(f"site-resample CI95 [{lo:+.4f},{hi:+.4f}]  null-centred p {p_null:.4g}  sign test {k_pos}/{n} p={p_sign:.4g}")
print("per-site p/Holm:", site_p_note)