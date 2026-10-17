#!/usr/bin/env python
"""Figure 3 — mechanism localization ladder (all values read from registry v4)."""
import csv, os, pathlib
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42   # embed TrueType, not Type 3
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(os.environ.get("PAPERM_ROOT", "/mnt/disk_8tb/codex/2609-wan/paperM/MAGP-AHGNN"))
OUT = ROOT / "manuscript/figures/fig3_mechanism.pdf"
reg = {r["experiment_id"]: r for r in csv.DictReader(open(ROOT / "result_registry_paperM_v4.csv"))}


def row(eid, label, kind):
    r = reg[eid]
    return dict(label=label, est=float(r["estimate"]),
                lo=float(r["ci_low"]), hi=float(r["ci_high"]), kind=kind)


ROWS = [
    row("full_sex_site_s666", "Full model (sex+site edges)", "ref"),
    row("mech_real_site_s666_maskTT", "No train\u2192test edges", "int"),
    row("P1b_supervision_masked_same-site_labels_masked", "Same-site supervision masked", "int"),
    row("p1c_budget_matched_s666", "Same-site kept, budget matched", "ctl"),
    row("P1a_head_single_transformer", "Single-channel head", "arch"),
    row("P1a_head_dual_identical", "Dual channel, identical edges", "arch"),
    row("parisot_full", "Canonical Parisot-GCN (topology only)", "arch"),
]
# `full_sex_site_s666` resolves to the latest row with that id; the registry holds
# one entry per id, so a plain dict is sufficient.

COL = {"ref": "#1f4e79", "int": "#c0392b", "ctl": "#2e7d32", "arch": "#7b1fa2"}
fig, ax = plt.subplots(figsize=(6.6, 3.2))
ys = list(range(len(ROWS)))[::-1]
ref = ROWS[0]["est"]
ax.axvline(ref, color="#1f4e79", lw=0.9, ls="--", zorder=1)
for y, r in zip(ys, ROWS):
    c = COL[r["kind"]]
    ax.plot([r["lo"], r["hi"]], [y, y], color=c, lw=1.6, solid_capstyle="round", zorder=2)
    ax.plot([r["est"]], [y], "o", color=c, ms=5.5, zorder=3)
    ax.text(r["est"] + 0.012, y, f"{r['est']:.3f}", va="center", ha="left",
            fontsize=7.5, color=c)
ax.set_yticks(ys)
ax.set_yticklabels([r["label"] for r in ROWS], fontsize=8)
ax.set_xlim(0.42, 1.03)
ax.set_xlabel("Pooled out-of-fold AUC (site-clustered 95\\% CI)", fontsize=8)
ax.tick_params(axis="x", labelsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="x", color="#dddddd", lw=0.6, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)
for r in ROWS:
    print(f"  {r['label'][:44]:46s} {r['est']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]")
