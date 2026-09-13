#!/usr/bin/env python3
"""Mechanism controls (decide HOW site-only ~0.95 arises), full-transductive,
site-only population edges, seed666.

  --feat const          : subject node features replaced by a constant -> if AUC
                          ~0.5, site topology alone is not a classifier.
  --feat shuffle_site   : real features, but each subject is assigned the imaging
                          of a random SAME-SITE subject (per fold, train side),
                          breaking subject imaging<->label while keeping site
                          structure/feature distribution -> if AUC ~0.5, the
                          model needs within-site subject imaging (it uses it).
  --feat real           : baseline (site-only graph, real features) ~0.947.
Run: PYTHONPATH=src $PY scripts/mechanism_controls.py --feat const|shuffle_site
"""
from __future__ import annotations
import os, argparse, csv, importlib.util, json, sys, time
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


def _edges_site(sex, site, train, cutoff=1.0):
    n = len(sex)
    aff = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if site[i] == site[j]:
                aff[i, j] += 1; aff[j, i] += 1
    ref = aff[train, :]; std = ref.std(0); std[std == 0] = 1.0
    aff = (aff - ref.mean(0)) / std
    e = np.argwhere(aff > cutoff); e = e[e[:, 0] < e[:, 1]]
    return e.T.astype(np.int64)


def _batched_x(adjs, fused, pc, sc, feat_mode, rng_site=None, N=None, device=None):
    """Return (x_all, edge_all, eattr_all, batch_all, adj_stack) for ALL subjects.
    feat_mode const: constant features (skip pca/scaler); shuffle_site: real PCA
    features with per-subject assignment permuted among same-site subjects."""
    from torch_geometric.utils import remove_self_loops
    global _SITE, _rng
    xs, es, eas, adjs_s, lens = [], [], [], [], []
    offset = 0
    for i in range(N):
        if feat_mode == "const":
            xn = np.full((adjs[i].shape[0], 32), 1.0, dtype=np.float32)
        else:
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


def _rewire_degree_preserving(edges, n, seed):
    """Degree-preserving double-edge-swap (networkx), real topology control vs the
    site-only graph."""
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(edges[0], edges[1]))
    m = G.number_of_edges()
    if m < 4:
        return edges
    G = nx.double_edge_swap(G, nswap=max(100, 5 * m), max_tries=max(1000, 50 * m), seed=int(seed))
    arr = np.array(sorted(G.edges()), dtype=np.int64).T
    return arr if arr.size else edges


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="real", choices=["real", "const", "shuffle_site"])
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--seed", type=int, default=666)
    ap.add_argument("--mask_tt", action="store_true", help="C: drop train<->test population edges (keep within-train & within-test)")
    ap.add_argument("--rewire", action="store_true", help="degree-preserving double-edge-swap rewiring of the site-only edge set")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    het = load_module("het_magp", WT / "src" / "cr_magp_ahgnn" / "models" / "heterogeneous_magp.py")
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from scipy.special import softmax
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

    OUT.mkdir(parents=True, exist_ok=True)
    npz = np.load(CACHE)
    valid = [str(x) for x in npz["subject_ids"]]; adjs = npz["adj"]; fused = npz["fused_x"]
    N = len(valid)
    pheno = {}
    with open(PHENO, newline="") as f:
        for row in csv.DictReader(f): pheno[row["SUB_ID"]] = row
    y = np.zeros(N)
    usite = np.unique([pheno[_norm(s)]["SITE_ID"] for s in valid]).tolist()
    site = np.zeros(N, dtype=int); sex = np.zeros(N, dtype=int)
    for i, s in enumerate(valid):
        p = pheno[_norm(s)]
        y[i] = 1 if int(p["DX_GROUP"]) == 1 else 0
        site[i] = usite.index(p["SITE_ID"]); sex[i] = int(p["SEX"])
    labels = torch.from_numpy(y.astype(np.int64)).to(device)
    skf = StratifiedKFold(10, shuffle=True, random_state=args.seed)
    cv = list(skf.split(np.zeros(N), y))
    tag = f"mech_{args.feat}_site_s{args.seed}" + ("_maskTT" if args.mask_tt else "") + ("_rewire" if args.rewire else "")
    run = {"feat": args.feat, "edge": "site", "folds": [], "started": time.time()}
    oof = np.zeros((N, 2)); oof_fold = -np.ones(N, dtype=int)
    for fold in range(10):
        tr_full, te = cv[fold]
        sss = StratifiedShuffleSplit(1, test_size=max(1, int(len(tr_full) * 0.1)), random_state=args.seed)
        tr_rel, va_rel = next(sss.split(np.zeros(len(tr_full)), y[tr_full]))
        tri, vai = tr_full[tr_rel], tr_full[va_rel]
        tri_list = [int(i) for i in tri]
        # per-fold site->subject features shuffle assignment for shuffle_site (applies to subject features globally this fold)
        perm_feat = None
        if args.feat == "shuffle_site":
            perm_feat = np.arange(N)
            for s_ in range(len(usite)):
                idx = np.where(site == s_)[0]
                if len(idx) > 1:
                    perm_feat[idx] = idx[np.random.permutation(len(idx))]
        st = np.vstack([fused[i] for i in tri_list])
        sc = StandardScaler().fit(st); pc = PCA(32, random_state=args.seed).fit(sc.transform(st))
        # feature ordering: for shuffle_site, use perm_feat to re-map fused lookups
        fused_use = fused[perm_feat] if perm_feat is not None else fused
        x_all, e_all, ea_all, b_all, a_all = _batched_x(adjs, fused_use, pc, sc, args.feat, N=N, device=device)
        enp = _edges_site(sex, site, np.array(tri_list))
        if args.rewire:
            enp = _rewire_degree_preserving(enp, N, args.seed * 100 + fold)
        if args.mask_tt:
            # drop any edge with one endpoint in test and the other not in test
            mask_te = np.zeros(N, dtype=bool); mask_te[te] = True
            u, v = enp
            keep = ~(mask_te[u] ^ mask_te[v])  # both-in-test OR both-not-in-test
            enp = enp[:, keep]
        e_t = torch.from_numpy(enp).long().to(device)
        model = het.HeterogeneousMAGP(het.MAGPConfig(device=str(device))).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
        best = -1.0; pct = 0; best_sd = None
        for ep in range(args.epochs):
            model.train(); opt.zero_grad()
            emb = model.subject_encoder(x_all, e_all, ea_all, b_all, a_all)
            lg = model.population_gnn(emb, e_t, torch.zeros((2, 0), dtype=torch.long, device=device))
            loss = _calibrated_loss(lg[tri_list], labels[tri_list], 0.1, 2)
            loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                embv = model.subject_encoder(x_all, e_all, ea_all, b_all, a_all)
                lv = model.population_gnn(embv, e_t, torch.zeros((2, 0), dtype=torch.long, device=device)).cpu().numpy()
            yv = y[vai]; lvv = lv[vai]
            sval = float((np.argmax(lvv, 1) == yv).mean()) + float(roc_auc_score(yv, softmax(lvv, 1)[:, 1]))
            if sval > best:
                best = sval; pct = 0
                best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                pct += 1
            if pct > args.patience:
                break
        model.load_state_dict(best_sd); model.eval()
        with torch.no_grad():
            embv = model.subject_encoder(x_all, e_all, ea_all, b_all, a_all)
            lgT = model.population_gnn(embv, e_t, torch.zeros((2, 0), dtype=torch.long, device=device)).cpu().numpy()
        lgt = lgT[te]; yt = y[te]; pred = np.argmax(lgt, 1)
        tn = np.sum((pred == 0) & (yt == 0)); fp = np.sum((pred == 1) & (yt == 0))
        fn = np.sum((pred == 0) & (yt == 1)); tp = np.sum((pred == 1) & (yt == 1))
        sen = float(tp / (tp + fn + 1e-10)); spe = float(tn / (tn + fp + 1e-10))
        auc_ = float(roc_auc_score(yt, softmax(lgt, 1)[:, 1]))
        run["folds"].append({"fold": fold, "acc": float((pred == yt).mean()), "auc": auc_,
                             "sen": sen, "spe": spe, "best_val": best})
        oof[te] = lgt; oof_fold[te] = fold
        print(f"[{tag} fold {fold}] TEST acc {run['folds'][-1]['acc']:.4f} auc {auc_:.4f}", flush=True)
    cov = oof_fold >= 0
    run["pooled_oof_auc"] = round(float(roc_auc_score(y[cov], softmax(oof[cov], 1)[:, 1])), 4)
    run["mean_auc"] = round(float(np.mean([f["auc"] for f in run["folds"]])), 4)
    (OUT / f"{tag}_report.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / f"{tag}_oof.npz", oof_logits=oof, oof_label=y, oof_fold=oof_fold,
                        subject_ids=np.array(valid, dtype="U32"))
    print(f"[{tag}] pooled_OOF_AUC {run['pooled_oof_auc']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
