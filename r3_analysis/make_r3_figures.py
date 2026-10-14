#!/usr/bin/env python3
"""Render the two figures requested by the external review, from saved results only (vector output).

F1  ON/OFF protocol schematic: T, S, R, C, V and what each arm supervises on.
F2  Per-site forest plot: per-seed effects, site macro, the site-resample population interval, and
    the sites whose fold counts are reduced by the eligibility rule.
Both are drawn from major1_site_level.csv / major1_population_inference.json; no new numbers.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

R3 = Path("/mnt/disk_8tb/codex/2609-wan/paperM/PAPERM_CMIG_R3_20261011")
OUT = R3 / "r3_figures"
OUT.mkdir(exist_ok=True)

rows = list(csv.DictReader((R3 / "major1_site_level.csv").open(encoding="utf-8")))
pop = json.loads((R3 / "major1_population_inference.json").read_text(encoding="utf-8"))

# ---------------- F1: protocol schematic ----------------
fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
boxes = {
    "T\ntarget-site test fold": (0.4, 3.1, 2.3, 1.2, "#dbe9f6"),
    "S\ntarget-site support": (0.4, 1.5, 2.3, 1.2, "#dff0d8"),
    "R\nother-site support": (3.2, 1.5, 2.3, 1.2, "#fce9d9"),
    "C\ncommon trainable": (6.0, 1.5, 2.3, 1.2, "#eeeeee"),
    "V\nfixed validation\n(no population edges)": (6.0, 3.1, 2.3, 1.2, "#f5e6f7"),
}
for label, (x, y, w, h, c) in boxes.items():
    ax.add_patch(Rectangle((x, y), w, h, facecolor=c, edgecolor="#444", lw=1.2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)
ax.text(0.4, 4.6, "Supervised set = S ∪ C ∪ R  (identical total N and class counts in both arms)",
        fontsize=9, fontweight="bold")
ax.text(0.4, 0.9, "Arm ON :  target site supplies S   |   Arm OFF :  target site supplies no support "
                  "(equal-size, class-matched support comes from other sites)", fontsize=9)
ax.text(0.4, 0.4, "Population-graph edges: same-site pairs among ALL nodes except V "
                  "(S ∪ R ∪ T ∪ C); V carries no population edges; the second channel receives "
                  "no edges, though its root-weight transform still acts",
        fontsize=8, style="italic")
fig.tight_layout()
for ext in ("pdf", "svg", "png"):
    fig.savefig(OUT / f"F1_onoff_protocol_schematic.{ext}", dpi=300)
plt.close(fig)

# ---------------- F2: per-site forest ----------------
sites = [r["site"] for r in rows]
per_seed = [json.loads(r["per_seed"]) for r in rows]
seed_keys = ["666", "2025", "2026"]
sym_csv = R3 / "SYMMETRISED_VS_CANONICAL.csv"
sym_map = {}
if sym_csv.exists():
    for r in csv.DictReader(sym_csv.open(encoding="utf-8")):
        sym_map[r["site"]] = float(r["symmetrised_site_macro"])
fig, ax = plt.subplots(figsize=(6.6, 5.4))
y = range(len(sites))
lo_band, hi_band = pop["site_resample_ci95"]
ax.axvspan(lo_band, hi_band, color="#cfe3f7", alpha=0.5, zorder=0,
           label=f"population site-resample 95% CI [{lo_band:+.3f}, {hi_band:+.3f}]")
ax.axvline(pop["population_macro"], color="#1f4e79", lw=1.6, zorder=1,
           label=f"population mean {pop['population_macro']:+.3f}")
ax.axvline(0, color="#888", lw=1.0, ls="--", zorder=1)
for i, r in enumerate(rows):
    v = [per_seed[i][s] for s in seed_keys if s in per_seed[i]]
    ax.scatter(v, [i] * len(v), s=14, color="#777", alpha=0.85, zorder=3,
               label="per-seed effect" if i == 0 else None)
    ax.plot([float(r["ci_lo"]), float(r["ci_hi"])], [i, i], color="#333", lw=1.4, zorder=2,
            label="within-site conditional 95% CI" if i == 0 else None)
    ax.plot([sum(v) / len(v)], [i], marker="D", ms=6, color="#c0392b", zorder=4,
            label="site macro, directed construction (primary)" if i == 0 else None)
    if r["site"] in sym_map:
        ax.plot([sym_map[r["site"]]], [i], marker="s", ms=5, mfc="none", mec="#1f7a4d", mew=1.4, zorder=5,
                label="site macro, symmetrised (parallel analysis)" if i == 0 else None)
ax.set_yticks(list(y)); ax.set_yticklabels(sites, fontsize=9)
ax.set_xlabel(r"$\Delta$AUC  (ON $-$ OFF)")
ax.set_title("Major 1: supervision-location effect by site, both edge constructions", fontsize=11)
partial = [r["site"] for r in rows if int(r["folds_used"]) < 5]
man = json.loads((R3 / "SPLIT_AND_MASK_MANIFEST.json").read_text(encoding="utf-8"))
infeasible = sorted({c["site"] for c in man["cells"] if not c["feasible"]})
ax.text(0.98, 0.02, f"fewer than 5 folds (eligibility rule): {', '.join(partial) if partial else 'none'}"
                    f"\nno feasible cell at all: {', '.join(infeasible) if infeasible else 'none'}"
                    "\nper-site intervals are conditional on the fixed model and seeds",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7, style="italic")
ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.9)
fig.tight_layout()
for ext in ("pdf", "svg", "png"):
    fig.savefig(OUT / f"F2_site_delta_auc_forest.{ext}", dpi=300)
plt.close(fig)

for f in sorted(OUT.glob("*")):
    print(f.name, f.stat().st_size)
print("sites:", len(sites), "| partial-fold sites:", partial)
