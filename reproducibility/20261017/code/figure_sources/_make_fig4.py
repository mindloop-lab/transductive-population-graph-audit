#!/usr/bin/env python
"""Figure 4 (v2): cross-architecture cohort performance including matched heads."""
import csv, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42   # embed TrueType, not Type 3
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt

ROOT = "/mnt/disk_8tb/codex/2609-wan/paperM/MAGP-AHGNN"
OUT = os.path.join(ROOT, "manuscript", "figures")
P1 = {r["setting"]: r for r in csv.DictReader(open(os.path.join(ROOT, "p1_results.csv")))}
REG = list(csv.DictReader(open(os.path.join(ROOT, "result_registry_paperM_v2.csv"))))
def reg_val(eid):
    for r in REG:
        if r["experiment_id"] == eid:
            lo = float(r["ci_low"]) if r["ci_low"] else None
            hi = float(r["ci_high"]) if r["ci_high"] else None
            return float(r["estimate"]), lo, hi
    return None, None, None

# rows: (label, est, lo, hi, color)
rows = [
  ("MAGP dual-channel head (target)", *[float(P1["magp_dual"][k]) for k in ("pooled_oof_auc", "ci_low", "ci_high")], "#1f4e79"),
  ("MAGP + single-channel TransformerConv", *[float(P1["single_transformer"][k]) for k in ("pooled_oof_auc", "ci_low", "ci_high")], "#7a9cc6"),
  ("MAGP + GAT head", *[float(P1["gat"][k]) for k in ("pooled_oof_auc", "ci_low", "ci_high")], "#7a9cc6"),
  ("MAGP + GCN head", *[float(P1["gcn"][k]) for k in ("pooled_oof_auc", "ci_low", "ci_high")], "#7a9cc6"),
  ("MAGP + MLP head (no graph)", *[float(P1["mlp"][k]) for k in ("pooled_oof_auc", "ci_low", "ci_high")], "#7a9cc6"),
  ("Parisot-GCN-C (topology-only)", *reg_val("parisot_full"), "#9e9e9e"),
  ("Parisot-GCN-C + site node-feat", *reg_val("parisot_sitefeat_full"), "#c9c9c9"),
]
img = [r for r in REG if r["experiment_id"] == "imaging_rf_cpac"][0]["estimate"]
y = np.arange(len(rows))[::-1]
fig, ax = plt.subplots(figsize=(6.6, 4.0))
for (lab, est, lo, hi, col), yy in zip(rows, y):
    xerr = [[est-lo], [hi-est]] if lo is not None and hi is not None else None
    ax.errorbar(est, yy, xerr=xerr, fmt="o", color=col, ecolor="0.6", ms=6, capsize=2)
    ax.text(max(hi or est, est)+0.004, yy, f"{est:.3f}", va="center", fontsize=7)
ax.axvline(0.5, ls="--", color="0.4", lw=1)
ax.axvline(float(img), ls=":", color="#b26b00", lw=1)
ax.text(float(img), len(rows)-0.6, f"imaging-only {float(img):.2f}", color="#b26b00", fontsize=6.5, rotation=90, va="top")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
ax.set_xlim(0.45, 1.0); ax.set_xlabel("cohort pooled out-of-fold AUC (95% CI)")
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig4_crossarch.pdf")); plt.savefig(os.path.join(OUT, "fig4_crossarch.png"), dpi=300); plt.close()
print("fig4 updated. rows:", [(r[0], r[1]) for r in rows])
