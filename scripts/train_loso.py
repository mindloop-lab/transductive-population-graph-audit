#!/usr/bin/env python3
"""Leave-one-site-out (LOSO) evaluation of the clean heterogeneous MAGP-AHGNN.

For each held-out site S: train on all other subjects (intra-train transductive
graph, sex+site edges among train, val held from train for early stop). At test,
held-out subjects of S are inserted as queries connected ONLY by cross-site
(same-sex) edges to the train graph - no same-site edges, no S-subject-to-S-
subject edges - giving a true unseen-site evaluation.

Compares to imaging-only RF LOSO (~0.653 pooled) and to the within-fold
transductive full model (~0.94) to test whether the population model retains any
cross-scanner value once site is fully held out.

Run: PYTHONPATH=src $PY scripts/train_loso.py
"""
from __future__ import annotations
import csv, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

WT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("ABIDE_CACHE", WT / "data_contract" / "cache_abide_data.npz"))
PHENO = Path(os.environ.get("ABIDE_PHENO", WT / "data_contract" / "Phenotypic_V1_0b_preprocessed1.csv"))
OUT = Path(os.environ.get("PAPERM_OUT", WT / "results" / "transductive_controls"))


def _norm(s): return str(int(s))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


def _calibrated_loss(logits, target, smoothing=0.1, n_classes=2):
    import torch.nn.functional as F
    onehot = torch.zeros_like(logits).scatter(1, target.long().unsqueeze(1), 1.0)
    smooth = onehot * (1 - smoothing) + (1 - onehot) * smoothing / (n_classes - 1)
    return F.kl_div(F.log_softmax(logits, dim=1), smooth, reduction="none").sum(1).mean()


def _intra_edges(sex, site, idx, cutoff=1.0):
    """Edges among subjects in idx (transductive, sex+site affinity normalized on idx)."""
    sex = sex[idx]; site = site[idx]; m = len(idx)
    aff = np.zeros((m, m))
    for lab in (site, sex):
        for i in range(m):
            for j in range(i + 1, m):
                if lab[i] == lab[j]:
                    aff[i, j] += 1; aff[j, i] += 1
    ref = aff; std = ref.std(0); std[std == 0] = 1.0
    aff = (aff - ref.mean(0)) / std
    e = np.argwhere(aff > cutoff); e = e[e[:, 0] < e[:, 1]]
    return e.T.astype(np.int64)  # local indices among idx


def _cross_edges(tr_sex, te_sex):
    """same-sex edges between train (local 0..T-1) and test (local T..T+K-1)."""
    T = len(tr_sex); K = len(te_sex)
    rows, cols = [], []
    for i in range(K):
        for j in range(T):
            if te_sex[i] == tr_sex[j]:
                rows.append(i + T); cols.append(j)  # test(i)->train(j)
    return np.stack([np.array(cols, dtype=np.int64), np.array(rows, dtype=np.int64)])


def _batched_for(adjs, fused, pc, sc, idx, device):
    from torch_geometric.utils import remove_self_loops
    xs, es, eas, adjs_s, lens = [], [], [], [], []
    offset = 0
    for i in idx:
        xn = pc.transform(sc.transform(fused[i])).astype(np.float32)
        a = np.abs(adjs[i]); r, c = np.where(a > 0.5)
        ei = torch.from_numpy(np.vstack((r, c)).astype(np.int64) + offset)
        ea = torch.from_numpy(a[r, c].astype(np.float32)); ei, ea = remove_self_loops(ei, ea)
        xs.append(torch.from_numpy(xn)); es.append(ei); eas.append(ea)
        adjs_s.append(np.nan_to_num(adjs[i]).astype(np.float32)); lens.append(xn.shape[0]); offset += xn.shape[0]
    x_all = torch.cat(xs, 0).to(device); edge_all = torch.cat(es, 1).to(device)
    ea_all = torch.cat(eas, 0).to(device)
    batch_all = torch.cat([torch.full((ln,), i, dtype=torch.long, device=device) for i, ln in enumerate(lens)], 0)
    return x_all, edge_all, ea_all, batch_all, torch.from_numpy(np.stack(adjs_s)).to(device)


def main() -> int:
    torch.manual_seed(666); np.random.seed(666)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    het = load_module("het_magp", WT / "src" / "cr_magp_ahgnn" / "models" / "heterogeneous_magp.py")
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.model_selection import StratifiedShuffleSplit
    from scipy.special import softmax
    from sklearn.metrics import roc_auc_score

    npz = np.load(CACHE) if "--cache" not in sys.argv else np.load(Path(sys.argv[sys.argv.index("--cache") + 1]))
    valid = [str(x) for x in npz["subject_ids"]]; adjs = npz["adj"]; fused = npz["fused_x"]
    N = len(valid)
    pheno = {}
    with open(PHENO, newline="") as f:
        for row in csv.DictReader(f): pheno[row["SUB_ID"]] = row
    y = np.zeros(N); site = np.zeros(N, dtype=int); sex = np.zeros(N, dtype=int)
    usite = sorted({pheno[_norm(s)]["SITE_ID"] for s in valid})
    for i, s in enumerate(valid):
        p = pheno[_norm(s)]
        y[i] = 1 if int(p["DX_GROUP"]) == 1 else 0
        site[i] = usite.index(p["SITE_ID"]); sex[i] = int(p["SEX"])
    OUT.mkdir(parents=True, exist_ok=True)
    per = {}
    allp = np.zeros(N); allfold = np.zeros(N, dtype=bool)
    for si in range(len(usite)):
        te = np.where(site == si)[0]
        if len(te) < 2:
            continue
        tr_all = np.where(site != si)[0]
        # val from train for early stop
        sss = StratifiedShuffleSplit(1, test_size=0.1, random_state=666)
        tr_i, va_i = next(sss.split(np.zeros(len(tr_all)), y[tr_all]))
        tri_g = tr_all[tr_i]; vai_g = tr_all[va_i]        # global inner-train / val
        st = np.vstack([fused[i] for i in tri_g])
        sc = StandardScaler().fit(st); pc = PCA(32, random_state=666).fit(sc.transform(st))
        # graph covers ALL train-site subjects (order = tr_all); PCA fitted on tri_g only
        x_all, e_all, ea_all, b_all, a_all = _batched_for(adjs, fused, pc, sc, tr_all.tolist(), device)
        x_te, e_te, ea_te, b_te, a_te = _batched_for(adjs, fused, pc, sc, te.tolist(), device)
        T = len(tr_all)
        pos_tr = np.searchsorted(tr_all, tri_g)            # position of inner-train in tr_all
        pos_va = np.searchsorted(tr_all, vai_g)            # position of val in tr_all
        # intra edges among all train-site subjects
        e_intra = _intra_edges(sex, site, tr_all)
        # train phase
        model = het.HeterogeneousMAGP(het.MAGPConfig(device=str(device))).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
        e_tr = torch.from_numpy(e_intra).long().to(device)
        lbl = torch.from_numpy(y.astype(np.int64)).to(device)
        best = -1.0; pct = 0; best_sd = None; pat = 150; epmax = 300
        for ep in range(epmax):
            model.train(); opt.zero_grad()
            emb = model.subject_encoder(x_all, e_all, ea_all, b_all, a_all)
            lg = model.population_gnn(emb, e_tr, torch.zeros((2, 0), dtype=torch.long, device=device))
            loss = _calibrated_loss(lg[pos_tr], lbl[tri_g], 0.1, 2)
            loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                embv = model.subject_encoder(x_all, e_all, ea_all, b_all, a_all)
                lv = model.population_gnn(embv, e_tr, torch.zeros((2, 0), dtype=torch.long, device=device)).cpu().numpy()
            lv_va = lv[pos_va]; yv_va = y[vai_g]
            sval = float((np.argmax(lv_va, 1) == yv_va).mean()) + float(roc_auc_score(yv_va, softmax(lv_va, 1)[:, 1]))
            if sval > best:
                best = sval; pct = 0
                best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                pct += 1
            if pct > pat:
                break
        model.load_state_dict(best_sd); model.eval()
        # eval: train embeddings (0..T-1, over tr_all) + held-out-site embeddings (T..),
        # edges = intra-train + cross-site test<->train (same-sex). No test-test edges.
        e_cross = _cross_edges(sex[tr_all], sex[te])
        e_cross_t = torch.from_numpy(e_cross).long().to(device)
        with torch.no_grad():
            emb_tr = model.subject_encoder(x_all, e_all, ea_all, b_all, a_all)
            emb_te = model.subject_encoder(x_te, e_te, ea_te, b_te, a_te)
        emb_c = torch.cat([emb_tr, emb_te], 0)
        e_all_t = torch.cat([e_tr, e_cross_t], 1)  # e_tr already local among 0..T-1
        with torch.no_grad():
            lg_c = model.population_gnn(emb_c, e_all_t, torch.zeros((2, 0), dtype=torch.long, device=device)).cpu().numpy()
        lg_te = lg_c[T:]
        p = softmax(lg_te, 1)[:, 1]
        auc_ = float(roc_auc_score(y[te], p))
        per[usite[si]] = {"n": int(len(te)), "n_asd": int(y[te].sum()), "auc": round(auc_, 4)}
        allp[te] = p; allfold[te] = True
        print(f"LOSO site {usite[si]:>12} n={len(te):3d} AUC {auc_:.3f}", flush=True)
    pooled = float(roc_auc_score(y[allfold], allp[allfold]))
    out = {"pooled_auc": round(pooled, 4), "per_site": per, "n_sites": len(per)}
    # --- prediction dump (additive; enables audit + site-clustered CIs) --------
    np.savez(OUT / "model_loso_predictions.npz",
             y=y[allfold].astype(int), p=allp[allfold].astype(float),
             subject_ids=np.asarray([str(v) for v in np.asarray(valid)[allfold]], dtype=object),
             site=np.asarray([usite[k] for k in site[allfold]], dtype=object),
             pipeline=np.asarray([os.environ.get("ABIDE_PIPELINE", "unspecified")]))
    (OUT / "model_loso.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("MODEL LOSO pooled AUC", round(pooled, 4), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
