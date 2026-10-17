#!/usr/bin/env python
"""Regenerate Fig 2-4 (+ Fig S2) per 0910-v2: no in-image figure numbers, Fig3 =
in-cohort vs LOSO scatter, Fig4 reference-line relabel, site-size -> supplement."""
import csv, os, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42   # embed TrueType, not Type 3
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = os.environ.get("PAPERM_ROOT", "/mnt/disk_8tb/codex/2609-wan/paperM/MAGP-AHGNN")
OUT = os.path.join(ROOT, "manuscript", "figures")
CTRL = os.path.join(os.environ.get("E1_DIR", os.path.join(ROOT, "evidence_snapshot")), "transductive_controls")
REG = list(csv.DictReader(open(os.path.join(ROOT, os.environ.get("REG_CSV", "result_registry_paperM_v4.csv")))))
def rg(eid, field="estimate"):
    for r in REG:
        if r["experiment_id"] == eid:
            v = r[field]; return float(v) if v not in ("", None) else None
def bysrc(sub, field="estimate"):
    for r in REG:
        if sub in r["prediction_file"]:
            v = r[field]; return float(v) if v not in ("", None) else None
P1 = {r["setting"]: r for r in csv.DictReader(open(os.path.join(ROOT, "p1_results.csv")))}

# ---------- Fig 2: forest (no title) ----------
order = [("full_sex_site_s666","Full (sex+site)","primary"), ("full_site_s666","Site-only edges","primary"),
         ("full_sex_s666","Sex-only edges","primary"), ("full_random_s666","Random edges","primary"),
         ("subject_only_sex_site_s666","Subject-only (no graph)","primary"),
         ("mech_shuffle_site_site_s666","B: shuffle imaging within site","mech"),
         ("mech_real_site_s666_maskTT","C: no train\u2192test edges","mech"),
         ("mech_const_site_s666","A: constant features","mech"),
         ("mech_real_site_s666_rewire","D: rewiring","mech"),
         ("perm_sex_site_s666","Label permutation (null)","null")]
# site-clustered CIs recomputed in _make_figs_paperM (reuse values already in registry v2 for CI? registry has subject CI).
import torch
from sklearn.metrics import roc_auc_score
pheno = {str(int(float(r["SUB_ID"]))): r["SITE_ID"].strip() for r in csv.DictReader(open(os.environ.get("ABIDE_PHENO", "/mnt/disk_8tb/data/abide_pcp_multiatlas_v1/Phenotypic_V1_0b_preprocessed1.csv")))}
rng = np.random.default_rng(5)
def clus(npz):
    d = np.load(npz, allow_pickle=True); y = np.asarray(d["oof_label"]).ravel().astype(int)
    z = np.asarray(d["oof_logits"]); z = z - z.max(1, keepdims=True); e = np.exp(z); p = (e/e.sum(1, keepdims=True))[:,1]
    ids = [str(int(float(s))) for s in np.asarray(d["subject_ids"]).ravel()]; st = np.array([pheno.get(i,"NA") for i in ids])
    sites = np.unique(st); vals=[]
    for _ in range(1500):
        idx = np.concatenate([np.flatnonzero(st==s) for s in rng.choice(sites,len(sites),True)])
        if y[idx].sum() in (0,len(idx)): continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return roc_auc_score(y,p), np.percentile(vals,2.5), np.percentile(vals,97.5)
est=[];lo=[];hi=[];labs=[];grp=[]
for eid,lab,g in order:
    src = os.path.join(CTRL, eid+"_oof.npz")
    a,l,h = clus(src); est.append(a); lo.append(l); hi.append(h); labs.append(lab); grp.append(g)
yy = np.arange(len(order))[::-1]
fig, ax = plt.subplots(figsize=(6.4, 4.3))
for (lab, e, l, h, g), y in zip(zip(labs,est,lo,hi,grp), yy):
    col = {"primary":"#1f4e79","mech":"#b26b00","null":"#7a7a7a"}[g]
    ax.errorbar(e, y, xerr=[[e-l],[h-e]], fmt="o", color=col, ecolor="0.6", capsize=2, ms=5)
ax.axvline(0.5, ls="--", color="0.4", lw=1)
ax.set_yticks(yy); ax.set_yticklabels(labs, fontsize=8)
ax.set_xlim(0.4, 1.0); ax.set_xlabel("pooled out-of-fold AUC (site-clustered 95% CI)")
for s in ("top","right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig2_forest.pdf")); plt.savefig(os.path.join(OUT,"fig2_forest.png"),dpi=300); plt.close()

# ---------- Fig 3: (a) per-site LOSO ; (b) in-cohort vs LOSO ----------
los = json.load(open(os.path.join(CTRL,"model_loso.json")))["per_site"]
sub = json.load(open(os.path.join(ROOT,"evidence_snapshot/subgroup_analysis_seed666.json")))["per_site"]
names = [n for n,_ in sorted(sub.items(), key=lambda kv:-kv[1]["n"])]
la = [los[n]["auc"] for n in names]; ia = [sub[n]["auc"] for n in names]
fig, ax = plt.subplots(1,2, figsize=(9.6,4.0))
o = np.argsort(la)
ax[0].barh(range(len(names)), [la[i] for i in o], color="#1f4e79")
ax[0].set_yticks(range(len(names))); ax[0].set_yticklabels([names[i] for i in o], fontsize=6)
ax[0].axvline(0.5, ls="--", color="0.4"); ax[0].set_xlabel("LOSO AUC (unseen site)"); ax[0].set_title("(a) per-site LOSO", fontsize=9)
ax[1].plot([0,1],[0,1], ls="--", color="0.6")
ax[1].scatter(ia, la, c="#c0504d", s=18)
ax[1].set_xlim(0.1,1.05); ax[1].set_ylim(0.1,1.05); ax[1].set_xlabel("in-cohort site AUC"); ax[1].set_ylabel("LOSO site AUC")
ax[1].set_title("(b) in-cohort vs LOSO (per site)", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig3_loso.pdf")); plt.savefig(os.path.join(OUT,"fig3_loso.png"),dpi=300); plt.close()

# ---------- Fig S2: site-size (supplement) ----------
man = json.load(open(os.path.join(ROOT,"code_pkg/data_contract/phase_minus1b_splits.json")))
raw = torch.load(os.environ.get("ABIDE_FIXTURE", "/mnt/disk_8tb/codex/2609-wan/del/CR-MAGP-AHGNN/artifacts/phase0d/formal_seed666_20260829_v1/raw_input/raw_fixture.pt"), map_location="cpu", weights_only=False)
sint = np.asarray(raw["site"]).astype(int)
unmatched = sorted(sub.items(), key=lambda kv:-kv[1]["n"])
name_of = {}
for s in set(sint.tolist()):
    n_s = int((sint == s).sum())
    for i,(nm,d) in enumerate(unmatched):
        if d["n"] == n_s:
            name_of[s] = nm; unmatched.pop(i); break
nst={};cnt={}
for f in man["folds"]:
    tr=set(f["optimization_train"]["indices"])
    for i in f["test"]["indices"]:
        nm=name_of[sint[i]]; nst[nm]=nst.get(nm,0)+sum(1 for j in tr if sint[j]==sint[i]); cnt[nm]=cnt.get(nm,0)+1
x=[nst[n]/max(1,cnt[n]) for n in names]; yv=ia
rho,p = spearmanr(x,yv)
fig, ax = plt.subplots(figsize=(3.6,3.4)); ax.scatter(x,yv,c="#c0504d",s=18)
ax.set_xlabel("mean same-site train neighbours"); ax.set_ylabel("in-cohort site AUC")
ax.set_title(f"rho={rho:.2f}, p={p:.2f}", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"figS2_sitesize.pdf")); plt.savefig(os.path.join(OUT,"figS2_sitesize.png"),dpi=300); plt.close()

# ---------- Fig 4: cross-arch (reference line relabelled) ----------
rows = [("MAGP dual-channel head (target)", float(P1["magp_dual"]["pooled_oof_auc"]), float(P1["magp_dual"]["ci_low"]), float(P1["magp_dual"]["ci_high"]), "#1f4e79"),
        ("MAGP dual-channel, identical edges (control)", float(P1["dual_identical"]["pooled_oof_auc"]), float(P1["dual_identical"]["ci_low"]), float(P1["dual_identical"]["ci_high"]), "#a6c8e0"),
        ("MAGP + single-channel TransformerConv", float(P1["single_transformer"]["pooled_oof_auc"]), float(P1["single_transformer"]["ci_low"]), float(P1["single_transformer"]["ci_high"]), "#7a9cc6"),
        ("MAGP + GAT head", float(P1["gat"]["pooled_oof_auc"]), float(P1["gat"]["ci_low"]), float(P1["gat"]["ci_high"]), "#7a9cc6"),
        ("MAGP + GCN head", float(P1["gcn"]["pooled_oof_auc"]), float(P1["gcn"]["ci_low"]), float(P1["gcn"]["ci_high"]), "#7a9cc6"),
        ("MAGP + MLP head (no graph)", float(P1["mlp"]["pooled_oof_auc"]), float(P1["mlp"]["ci_low"]), float(P1["mlp"]["ci_high"]), "#7a9cc6"),
        ("Parisot-GCN-C (topology-only)", rg("parisot_full"), rg("parisot_full","ci_low"), rg("parisot_full","ci_high"), "#9e9e9e"),
        ("Parisot-GCN-C + site node-feat", rg("parisot_sitefeat_full"), rg("parisot_sitefeat_full","ci_low"), rg("parisot_sitefeat_full","ci_high"), "#c9c9c9")]
img = rg("imaging_rf_cpac")
yy = np.arange(len(rows))[::-1]
fig, ax = plt.subplots(figsize=(6.6,4.0))
for (lab,e,l,h,col), y in zip(rows, yy):
    xerr = [[e-l],[h-e]] if l is not None and h is not None else None
    ax.errorbar(e,y,xerr=xerr,fmt="o",color=col,ecolor="0.6",ms=6,capsize=2)
    ax.text(max(h or e,e)+0.004,y,f"{e:.3f}",va="center",fontsize=7)
ax.axvline(0.5, ls="--", color="0.4", lw=1)
ax.axvline(img, ls=":", color="#b26b00", lw=1.2)
ax.text(img+0.003, len(rows)-0.7, f"classical imaging reference (RF, {img:.2f})", color="#b26b00", fontsize=6.5, rotation=90, va="top")
ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
ax.set_xlim(0.45,1.0); ax.set_xlabel("cohort pooled out-of-fold AUC (95% CI)")
for s in ("top","right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig4_crossarch.pdf")); plt.savefig(os.path.join(OUT,"fig4_crossarch.png"),dpi=300); plt.close()
print("regenerated fig2_forest, fig3_loso (in-cohort vs LOSO), figS2_sitesize, fig4_crossarch (no in-image titles).")
print("fig3 spearman (site-size):", round(rho,3), round(p,3))
