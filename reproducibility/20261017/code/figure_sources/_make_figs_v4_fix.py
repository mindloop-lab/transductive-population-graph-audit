#!/usr/bin/env python
"""Correct Fig 3(b) and Fig S2 to use registry-consistent (post-E1) site-level values.

Problems fixed:
  * Fig S2's y-axis (in-cohort site AUC) came from
    evidence_snapshot/subgroup_analysis_seed666.json — a pre-E1 source — while the
    table reports post-E1 values;
  * the same pre-E1 source fed Fig 3(b)'s x-axis;
  * site names were mapped by matching subject counts (fragile); the fixture actually
    carries `subject_ids`, so the phenotype table gives an exact mapping.
Both figures are regenerated here, and Spearman and Pearson are reported for Fig S2.
"""
import csv, json, os, pathlib
import numpy as np, torch
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42   # embed TrueType, not Type 3
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(os.environ.get("PAPERM_ROOT", "/mnt/disk_8tb/codex/2609-wan/paperM/MAGP-AHGNN"))
OUT = ROOT / "manuscript/figures"
E1 = ROOT / os.environ.get("E1_DIR", "e1_out")
PH = os.environ.get("ABIDE_PHENO", "/mnt/disk_8tb/data/abide_pcp/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv")
FIX = os.environ.get("ABIDE_FIXTURE",
                     "/mnt/disk_8tb/codex/2609-wan/del/CR-MAGP-AHGNN/artifacts/phase0d/formal_seed666_20260829_v1/raw_input/raw_fixture.pt")

ph = {str(int(float(r["SUB_ID"]))): r["SITE_ID"].strip() for r in csv.DictReader(open(PH))}

# ---- post-E1 in-cohort site AUC from the frozen OOF predictions ----------------
oof = np.load(E1 / "transductive_controls/full_sex_site_s666_oof.npz", allow_pickle=True)
y = np.asarray(oof["oof_label"]).ravel().astype(int)
z = np.asarray(oof["oof_logits"])
pr = (np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True))[:, 1]
ids = [str(int(float(s))) for s in np.asarray(oof["subject_ids"]).ravel()]
site = np.array([ph.get(i, "NA") for i in ids])
ia = {}
for s in np.unique(site):
    m = site == s
    if len(set(y[m])) > 1:
        ia[s] = float(roc_auc_score(y[m], pr[m]))
print(f"post-E1 in-cohort site AUC computed for {len(ia)} sites")

# ---- post-E1 LOSO per site ----------------------------------------------------
los = json.load(open(E1 / "transductive_controls/model_loso.json"))["per_site"]
la = {k: v["auc"] if isinstance(v, dict) else float(v) for k, v in los.items()}
print("LOSO sites:", len(la))

common = sorted(set(ia) & set(la), key=lambda n: -len(np.flatnonzero(site == n)))
print("sites with both:", len(common))

# ---- same-site training neighbours, exact site mapping ------------------------
raw = torch.load(FIX, map_location="cpu", weights_only=False)
sint = np.asarray(raw["site"]).astype(int)
sids = [str(int(float(s))) for s in np.asarray(raw["subject_ids"]).ravel()]
sub_site = np.array([ph.get(i, "NA") for i in sids])
man = json.load(open(ROOT / "code_pkg/data_contract/phase_minus1b_splits.json"))
nst, cnt = {}, {}
for f in man["folds"]:
    tr = set(f["optimization_train"]["indices"])
    for i in f["test"]["indices"]:
        sn = sub_site[i]
        nst[sn] = nst.get(sn, 0) + sum(1 for j in tr if sub_site[j] == sn)
        cnt[sn] = cnt.get(sn, 0) + 1
x = [nst[s] / max(1, cnt[s]) for s in common]
yv = [ia[s] for s in common]

rho, prho = spearmanr(x, yv)
r, pp = pearsonr(x, yv)
print(f"\nFigS2 (post-E1, exact site mapping, n={len(common)}):")
print(f"  Spearman rho = {rho:.3f}  p = {prho:.4f}")
print(f"  Pearson  r   = {r:.3f}  p = {pp:.4f}")

# ---- regenerate Fig S2 -------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.6, 3.4))
ax.scatter(x, yv, c="#c0504d", s=18)
ax.set_xlabel("mean same-site train neighbours", fontsize=8)
ax.set_ylabel("in-cohort site AUC", fontsize=8)
ax.tick_params(labelsize=7)
plt.tight_layout(); plt.savefig(OUT / "figS2_sitesize.pdf"); plt.savefig(OUT / "figS2_sitesize.png", dpi=300); plt.close()
print("regenerated figS2_sitesize.pdf")

# ---- regenerate Fig 3 with the post-E1 in-cohort values ----------------------
names = common
fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.0))
o = np.argsort([la[n] for n in names])
ax[0].barh(range(len(names)), [la[names[i]] for i in o], color="#1f4e79")
ax[0].set_yticks(range(len(names))); ax[0].set_yticklabels([names[i] for i in o], fontsize=6)
ax[0].axvline(0.5, ls="--", color="0.4"); ax[0].set_xlabel("LOSO AUC (unseen site)")
ax[1].plot([0, 1], [0, 1], ls="--", color="0.6")
ax[1].scatter([ia[n] for n in names], [la[n] for n in names], c="#c0504d", s=18)
ax[1].set_xlim(0.1, 1.05); ax[1].set_ylim(0.1, 1.05)
ax[1].set_xlabel("in-cohort site AUC"); ax[1].set_ylabel("LOSO site AUC")
plt.tight_layout(); plt.savefig(OUT / "fig3_loso.pdf"); plt.savefig(OUT / "fig3_loso.png", dpi=300); plt.close()
print("regenerated fig3_loso.pdf (both panels now post-E1)")
print("\nfigS2 caption should read: n=%d, Spearman rho=%.2f (p=%.2f)" % (len(common), rho, prho))
