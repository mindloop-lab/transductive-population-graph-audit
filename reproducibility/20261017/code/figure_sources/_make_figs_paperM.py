#!/usr/bin/env python
"""Generate Paper M results figures from result_registry_paperM_v2.csv (+ splits
for same-site-train-neighbor counts). Reproducible, code-generated (no AI)."""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42   # embed TrueType, not Type 3
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = "/mnt/disk_8tb/codex/2609-wan/paperM/MAGP-AHGNN"
OUT = os.path.join(ROOT, "manuscript", "figures")
os.makedirs(OUT, exist_ok=True)
REG = {r["experiment_id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "result_registry_paperM_v2.csv")))}

def f(k, field="estimate"):
    v = REG[k][field]; return float(v) if v not in ("", None) else None

# ---------- Figure 2: forest plot ----------
order = [
 ("full_sex_site_s666", "Full (sex+site)"),
 ("full_site_s666", "Site-only edges"),
 ("mech_shuffle_site_site_s666", "B: shuffle imaging within site"),
 ("full_sex_s666", "Sex-only edges"),
 ("full_random_s666", "Random edges"),
 ("subject_only_sex_site_s666", "Subject-only (no graph)"),
 ("mech_real_site_s666_maskTT", "C: no train\u2192test edges"),
 ("mech_const_site_s666", "A: constant features"),
 ("mech_real_site_s666_rewire", "D: degree-preserving rewiring"),
 ("perm_sex_site_s666", "Label permutation (null)"),
]
est = [f(k) for k, _ in order]; lo = [f(k, "ci_low") for k, _ in order]; hi = [f(k, "ci_high") for k, _ in order]
labels = [n for _, n in order]

# site-clustered bootstrap CI (resample the 20 sites with replacement)
import torch, csv as _csv
_pheno = {}
for r in _csv.DictReader(open("/mnt/disk_8tb/data/abide_pcp_multiatlas_v1/Phenotypic_V1_0b_preprocessed1.csv")):
    _pheno[str(int(float(r["SUB_ID"])))] = r["SITE_ID"].strip()
def cluster_ci(npz_path, nboot=1500, seed=0):
    d = np.load(npz_path, allow_pickle=True)
    y = np.asarray(d["oof_label"]).ravel().astype(int)
    zz = np.asarray(d["oof_logits"])
    if zz.shape[1] == 2:
        zz = zz - zz.max(1, keepdims=True); e = np.exp(zz); p = (e/e.sum(1, keepdims=True))[:, 1]
    else:
        p = np.asarray(d["oof_prob"]).ravel()
    ids = [str(s) for s in np.asarray(d["subject_ids"]).ravel()]
    st = np.array([_pheno.get(str(int(float(i))), "NA") for i in ids])
    sites = np.unique(st); rng = np.random.default_rng(seed); vals = []
    for _ in range(nboot):
        pick = rng.choice(sites, len(sites), replace=True)
        idx = np.concatenate([np.flatnonzero(st == s) for s in pick])
        if y[idx].sum() in (0, len(idx)): continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)

from sklearn.metrics import roc_auc_score
src_of = {
 "full_sex_site_s666": "transductive_controls/full_sex_site_s666_oof.npz",
 "full_site_s666": "transductive_controls/full_site_s666_oof.npz",
 "mech_shuffle_site_site_s666": "transductive_controls/mech_shuffle_site_site_s666_oof.npz",
 "full_sex_s666": "transductive_controls/full_sex_s666_oof.npz",
 "full_random_s666": "transductive_controls/full_random_s666_oof.npz",
 "subject_only_sex_site_s666": "transductive_controls/subject_only_sex_site_s666_oof.npz",
 "mech_real_site_s666_maskTT": "transductive_controls/mech_real_site_s666_maskTT_oof.npz",
 "mech_const_site_s666": "transductive_controls/mech_const_site_s666_oof.npz",
 "mech_real_site_s666_rewire": "transductive_controls/mech_real_site_s666_rewire_oof.npz",
 "perm_sex_site_s666": "transductive_controls/perm_sex_site_s666_oof.npz",
}
ES = os.path.join(ROOT, "evidence_snapshot")
for k in list(order):
    kk = k[0]
    if kk in src_of:
        c_lo, c_hi = cluster_ci(os.path.join(ES, src_of[kk]))
        lo[order.index(k)] = float(c_lo); hi[order.index(k)] = float(c_hi)
        print(f"{kk:30s} subj[{f(kk,'ci_low'):.3f},{f(kk,'ci_high'):.3f}] site-clustered[{c_lo:.3f},{c_hi:.3f}]")

y = np.arange(len(order))[::-1]
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.hlines(np.arange(len(order)), 0.42, 1.0, color="0.9", lw=0.6, zorder=0)
ax.errorbar(est, y, xerr=[np.array(est)-np.array(lo), np.array(hi)-np.array(est)],
            fmt="o", color="#1f4e79", ecolor="#7aa6cc", capsize=2, ms=5, lw=1.2)
ax.axvline(0.5, ls="--", color="0.4", lw=1)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
ax.set_xlim(0.42, 1.0); ax.set_xlabel("Pooled out-of-fold AUC (95% CI, site-clustered bootstrap)")
ax.set_title("Fig. 2  Cohort performance by setting and mechanism intervention", fontsize=9)
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig2_forest.pdf")); plt.savefig(os.path.join(OUT, "fig2_forest.png"), dpi=300); plt.close()

# ---------- Figure 3: LOSO per-site + site-size ----------
los = json.load(open(os.path.join(ROOT, "evidence_snapshot/transductive_controls/model_loso.json")))["per_site"]
sub = json.load(open(os.path.join(ROOT, "evidence_snapshot/subgroup_analysis_seed666.json")))["per_site"]
# same-site train neighbour count per site (from splits manifest)
man = json.load(open(os.path.join(ROOT, "code_pkg/data_contract/phase_minus1b_splits.json")))
import torch
raw = torch.load("/mnt/disk_8tb/codex/2609-wan/del/CR-MAGP-AHGNN/artifacts/phase0d/formal_seed666_20260829_v1/raw_input/raw_fixture.pt", map_location="cpu", weights_only=False)
site = np.asarray(raw["site"]).astype(int)
sites = sorted(set(site.tolist()))
# map site int -> name via site_analysis.json per_site order? use counts to match: build name via subgroup keys by size
sub_items = sorted(sub.items(), key=lambda kv: -kv[1]["n"])
name_of_int = {}
# heuristic: match each site int (by count) to subgroup name (by n) uniquely
unmatched = list(sub_items)
for s in sites:
    n = int((site == s).sum())
    for i, (nm, d) in enumerate(unmatched):
        if d["n"] == n:
            name_of_int[s] = nm; unmatched.pop(i); break
nsts = {name_of_int[s]: 0.0 for s in sites}
cnt = {name_of_int[s]: 0 for s in sites}
for fold in man["folds"]:
    tr = set(fold["optimization_train"]["indices"]); te = fold["test"]["indices"]
    for i in te:
        same = sum(1 for j in tr if site[j] == site[i])
        nsts[name_of_int[site[i]]] += same; cnt[name_of_int[site[i]]] += 1
mean_nst = {k: (nsts[k]/cnt[k] if cnt[k] else 0) for k in nsts}
names = [nm for nm, _ in sub_items]
los_all = [los[nm]["auc"] for nm in names]; sub_all = [sub[nm]["auc"] for nm in names]
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
o = np.argsort(los_all)
axes[0].barh(range(len(names)), [los_all[i] for i in o], color="#1f4e79")
axes[0].set_yticks(range(len(names))); axes[0].set_yticklabels([names[i] for i in o], fontsize=6)
axes[0].axvline(0.5, ls="--", color="0.4"); axes[0].set_xlabel("LOSO AUC (unseen site)"); axes[0].set_title("(a) per-site LOSO", fontsize=9)
x = [mean_nst[nm] for nm in names]; yv = sub_all
rho, p = spearmanr(x, yv)
axes[1].scatter(x, yv, c="#c0504d", s=18)
axes[1].set_xlabel("mean same-site train neighbours"); axes[1].set_ylabel("in-cohort site AUC")
axes[1].set_title(f"(b) site-size vs cohort gain  (Spearman rho={rho:.2f}, p={p:.3f})", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig3_loso.pdf")); plt.savefig(os.path.join(OUT, "fig3_loso.png"), dpi=300); plt.close()
print(f"fig3 spearman rho={rho:.3f} p={p:.4f} (n_sites={len(names)})")

# ---------- Figure 4: cross-architecture ----------
def row(k): return f(k), f(k, "ci_low"), f(k, "ci_high")
models = [
 ("Hetero MAGP\n(site-aware)", row("full_sex_site_s666"), row("full_site_s666"), (f("magp_loso_cpac"), None, None)),
 ("Parisot-GCN\n(topology-only)", row("parisot_full"), row("parisot_site_only"), row("parisot_full") if False else (f("parisot_full", "estimate"), None, None)),
 ("Parisot + site\nnode feature", row("parisot_sitefeat_full"), row("parisot_sitefeat_site_only"), (None, None, None)),
]
# use LOSO parisot from loso rows
par_loso = f("parisot_site_only", "estimate") if "parisot_site_only" in REG else None
# fix: parisot loso rows share id 'parisot_full'/'parisot_site_only' collide with cohort; registry overwrote -> reread by source
by_src = {r["prediction_file"]: r for r in csv.DictReader(open(os.path.join(ROOT, "result_registry_paperM_v2.csv")))}
def srcval(substr, field="estimate"):
    for k, r in by_src.items():
        if substr in k: 
            v = r[field]; return float(v) if v not in ("", None) else None
    return None
groups = ["Cohort (full)", "Cohort (site-only)", "LOSO (unseen site)"]
data = {
 "Hetero MAGP": [f("full_sex_site_s666"), f("full_site_s666"), f("magp_loso_cpac")],
 "Parisot-GCN": [srcval("cohort/full_seed666"), srcval("cohort/site_only_seed666"), srcval("loso/full_seed666")],
 "Parisot+site feat": [srcval("sitefeat_full_seed666"), srcval("sitefeat_site_only_seed666"), None],
}
x = np.arange(len(groups)); w = 0.26
fig, ax = plt.subplots(figsize=(6.6, 3.8))
for i, (m, vals) in enumerate(data.items()):
    ax.bar(x + (i-1)*w, [v if v else 0 for v in vals], w, label=m,
           color=["#1f4e79", "#9e9e9e", "#c0504d"][i])
ax.axhline(0.5, ls="--", color="0.4"); ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=8)
ax.set_ylim(0.4, 1.0); ax.set_ylabel("pooled OOF AUC"); ax.legend(fontsize=7)
ax.set_title("Fig. 4  Cross-architecture scoping (site-aware vs topology-only)", fontsize=9)
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig4_crossarch.pdf")); plt.savefig(os.path.join(OUT, "fig4_crossarch.png"), dpi=300); plt.close()
print("figures written to", OUT)
print("mgp:", data["Hetero MAGP"]); print("par:", data["Parisot-GCN"]); print("psf:", data["Parisot+site feat"])
