#!/usr/bin/env python
"""Regenerate figS3 (in-cohort reliability diagram) from the frozen seed-666 OOF
predictions, with TrueType fonts. Also verifies the caption's Brier/ECE numbers.
Definition: 10 equal-width probability bins on the in-cohort OOF probabilities;
ECE = sum over bins of (n_b/n) * |mean(p_b) - mean(y_b)|. Nothing assumes the
original (lost) generator.
"""
import os, pathlib
import numpy as np
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42   # embed TrueType, not Type 3
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

ROOT = pathlib.Path("/mnt/disk_8tb/codex/2609-wan/paperM/MAGP-AHGNN")
E1 = ROOT / os.environ.get("E1_DIR", "e1_out")
FIG = ROOT / "manuscript/figures"

z = np.load(E1 / "transductive_controls/full_sex_site_s666_oof.npz", allow_pickle=True)
zz = np.asarray(z["oof_logits"])
y = np.asarray(z["oof_label"]).ravel().astype(int)
e = np.exp(zz - zz.max(1, keepdims=True))
p = (e / e.sum(1, keepdims=True))[:, 1]

brier = float(np.mean((p - y) ** 2))
bins = np.linspace(0.0, 1.0, 11)
idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
conf, acc, cnt = [], [], []
for b in range(10):
    m = idx == b
    if m.sum() == 0:
        continue
    conf.append(float(p[m].mean())); acc.append(float(y[m].mean())); cnt.append(int(m.sum()))
conf, acc, cnt = np.array(conf), np.array(acc), np.array(cnt)
ece = float(np.sum((cnt / cnt.sum()) * np.abs(conf - acc)))
print(f"recomputed from the frozen OOF: Brier = {brier:.3f}   ECE(10 bins) = {ece:.3f}")
print(f"caption currently says:          Brier = 0.092   ECE(10 bins) = 0.063")
print(f"bins used: {len(conf)}  (n = {int(cnt.sum())})")

fig, ax = plt.subplots(figsize=(3.4, 3.4))
ax.plot([0, 1], [0, 1], ls="--", color="0.6", lw=1)
ax.plot(conf, acc, "o-", color="#1f4e79", ms=4, lw=1.2)
ax.set_xlabel("predicted probability", fontsize=8)
ax.set_ylabel("observed frequency", fontsize=8)
ax.set_title(f"seed 666 in-cohort\nBrier {brier:.3f}, ECE {ece:.3f}", fontsize=8)
ax.tick_params(labelsize=7)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "figS3_reliability.pdf")
fig.savefig(FIG / "figS3_reliability.png", dpi=300)
print("wrote", FIG / "figS3_reliability.pdf")
