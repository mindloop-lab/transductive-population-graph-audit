#!/usr/bin/env python3
"""
Full-transductive main experiment, FAST + corrected end-to-end training.

Correctness fix vs the first 10-fold run: subject-encoder gradients were
previously blocked by torch.no_grad(), freezing the representation learner.
This version trains the WHOLE model end-to-end (subject_encoder + population
head) every epoch, matching the recovered protocol. Speedup: subject graphs are
pre-concatenated once per fold into a single batched graph
(encode_subjects_batched, verified == per-subject loop to 8e-7) so each epoch is
one batched forward instead of 871 python-loop forwards.

Protocol unchanged: fold-local PCA on optimization-train; transductive edges
(train-normalized sex+site affinity, all-subject nodes incl. test); Adam lr=1e-3
wd=5e-5; CalibratedLoss(label smoothing 0.1); <= max_epochs; early stop
(patience) by val acc+auc. Crash-safe per-fold writes under artifacts/.

Run:  PYTHONPATH=src $PY scripts/train_transductive_10fold_fast.py --epochs 400 --seed 666
"""
from __future__ import annotations
import os, argparse, csv, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import torch

WT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("ABIDE_CACHE", WT / "data_contract" / "cache_abide_data.npz"))
PHENO = Path(os.environ.get("ABIDE_PHENO", WT / "data_contract" / "Phenotypic_V1_0b_preprocessed1.csv"))
OUT = Path(os.environ.get("PAPERM_OUT", WT / "results" / "transductive_main"))


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


def _edges(sex, site, train, cutoff=1.0):
    n = len(sex)
    aff = np.zeros((n, n))
    for lab in (site, sex):
        for i in range(n):
            for j in range(i + 1, n):
                if lab[i] == lab[j]:
                    aff[i, j] += 1; aff[j, i] += 1
    ref = aff[train, :]; std = ref.std(0); std[std == 0] = 1.0
    aff = (aff - ref.mean(0)) / std
    e = np.argwhere(aff > cutoff); e = e[e[:, 0] < e[:, 1]]
    same = e[sex[e[:, 0]] == sex[e[:, 1]]]; diff = e[sex[e[:, 0]] != sex[e[:, 1]]]
    return same.T.astype(np.int64), diff.T.astype(np.int64)


def _batched(adjs, fused, pc, sc, device, threshold=0.5):
    """Concatenate all subjects into one batched graph (verified == per-subject)."""
    from torch_geometric.data import Data
    from torch_geometric.utils import remove_self_loops
    N = len(adjs)
    xs, es, eas, adjs_s, lens = [], [], [], [], []
    offset = 0
    for i in range(N):
        xn = pc.transform(sc.transform(fused[i])).astype(np.float32)
        a = np.abs(adjs[i]); r, c = np.where(a > threshold)
        ei = torch.from_numpy(np.vstack((r, c)).astype(np.int64) + offset)
        ea = torch.from_numpy(a[r, c].astype(np.float32))
        ei, ea = remove_self_loops(ei, ea)
        xs.append(torch.from_numpy(xn)); es.append(ei); eas.append(ea)
        adjs_s.append(np.nan_to_num(adjs[i]).astype(np.float32))
        lens.append(xn.shape[0]); offset += xn.shape[0]
    x_all = torch.cat(xs, 0).to(device)
    edge_all = torch.cat(es, 1).to(device)
    edge_attr_all = torch.cat(eas, 0).to(device)
    batch_all = torch.cat([torch.full((ln,), i, dtype=torch.long, device=device)
                           for i, ln in enumerate(lens)], 0)
    adj_stack = torch.from_numpy(np.stack(adjs_s)).to(device)
    return x_all, edge_all, edge_attr_all, batch_all, adj_stack


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=666)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--ckpt-out", type=str, default="", help="dir to save best per-fold .pt checkpoints (main-model archive)")
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
    run = {"seed": args.seed, "protocol": "full_transductive", "epochs": args.epochs, "end_to_end": True,
           "batched_encode": True, "folds": [], "started": time.time()}
    oof_logits = np.zeros((N, 2)); oof_fold = -np.ones(N, dtype=int)

    for fold in range(10):
        t0 = time.time(); tr_full, te = cv[fold]
        sss = StratifiedShuffleSplit(1, test_size=max(1, int(len(tr_full) * 0.1)), random_state=args.seed)
        tr_rel, va_rel = next(sss.split(np.zeros(len(tr_full)), y[tr_full]))
        tri, vai = tr_full[tr_rel], tr_full[va_rel]
        tri_list = [int(i) for i in tri]; vai_list = [int(i) for i in vai]
        st = np.vstack([fused[i] for i in tri_list])
        sc = StandardScaler().fit(st); pc = PCA(32, random_state=args.seed).fit(sc.transform(st))
        x_all, edge_all, eattr_all, batch_all, adj_stack = _batched(adjs, fused, pc, sc, device)
        same_np, diff_np = _edges(sex, site, np.array(tri_list))
        same = torch.from_numpy(same_np).long().to(device); diff = torch.from_numpy(diff_np).long().to(device)
        model = het.HeterogeneousMAGP(het.MAGPConfig(device=str(device))).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
        best = -1.0; pct = 0; best_sd = None
        for ep in range(args.epochs):
            model.train(); opt.zero_grad()
            emb = model.subject_encoder(x_all, edge_all, eattr_all, batch_all, adj_stack)
            lg = model.population_gnn(emb, same, diff)
            loss = _calibrated_loss(lg[tri_list], labels[tri_list], 0.1, 2)
            loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                embe = model.subject_encoder(x_all, edge_all, eattr_all, batch_all, adj_stack)
                lv = model.population_gnn(embe, same, diff).cpu().numpy()[vai_list]
            yv = y[vai_list]
            sval = float((np.argmax(lv, 1) == yv).mean()) + float(roc_auc_score(yv, softmax(lv, 1)[:, 1]))
            if sval > best:
                best = sval; pct = 0
                best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                pct += 1
            if pct > args.patience:
                print(f"[fold {fold}] early stop @ {ep} (best val {best:.3f})", flush=True); break
        model.load_state_dict(best_sd); model.eval()
        with torch.no_grad():
            embe = model.subject_encoder(x_all, edge_all, eattr_all, batch_all, adj_stack)
            lgT = model.population_gnn(embe, same, diff).cpu().numpy()
        lgt = lgT[te]; yt = y[te]; pred = np.argmax(lgt, 1)
        tn = np.sum((pred == 0) & (yt == 0)); fp = np.sum((pred == 1) & (yt == 0))
        fn = np.sum((pred == 0) & (yt == 1)); tp = np.sum((pred == 1) & (yt == 1))
        sen = float(tp / (tp + fn + 1e-10)); spe = float(tn / (tn + fp + 1e-10))
        auc_ = float(roc_auc_score(yt, softmax(lgt, 1)[:, 1]))
        f1 = float(precision_recall_fscore_support(yt, pred, average="binary", zero_division=0)[2])
        rec = {"fold": fold, "acc": float((pred == yt).mean()), "sen": sen, "spe": spe,
               "auc": auc_, "f1": f1, "ba": float((sen + spe) / 2), "best_val": best,
               "sec": round(time.time() - t0, 1)}
        if args.ckpt_out:
            ckdir = Path(args.ckpt_out)
            ckdir.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": best_sd, "fold": fold, "seed": args.seed,
                        "protocol": "full_transductive", "acc": rec["acc"], "auc": auc_},
                       ckdir / f"magp_hetero_transductive_seed{args.seed}_fold{fold}.pt")
        run["folds"].append(rec); oof_logits[te] = lgt; oof_fold[te] = fold
        print(f"[fold {fold}] TEST acc {rec['acc']:.4f} auc {auc_:.4f} sen {sen:.4f} spe {spe:.4f} "
              f"f1 {f1:.4f} ({rec['sec']}s)", flush=True)
        (OUT / f"seed{args.seed}_fast_partial.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
        np.savez_compressed(OUT / f"seed{args.seed}_fast_oof_partial.npz", oof_logits=oof_logits,
                            oof_label=y, oof_fold=oof_fold, subject_ids=np.array(valid, dtype="U32"))
    mean = {k: float(np.mean([f[k] for f in run["folds"]])) for k in ("acc", "sen", "spe", "auc", "f1", "ba")}
    std = {k: float(np.std([f[k] for f in run["folds"]])) for k in ("acc", "sen", "spe", "auc", "f1", "ba")}
    cov = oof_fold >= 0
    run["mean"] = {k: round(v, 4) for k, v in mean.items()}; run["std"] = {k: round(v, 4) for k, v in std.items()}
    run["pooled_oof_auc"] = round(float(roc_auc_score(y[cov], softmax(oof_logits[cov], 1)[:, 1])), 4)
    run["oof_coverage"] = int(cov.sum()); run["finished"] = time.time()
    (OUT / f"seed{args.seed}_fast_report.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / f"seed{args.seed}_fast_oof.npz", oof_logits=oof_logits, oof_label=y,
                        oof_fold=oof_fold, subject_ids=np.array(valid, dtype="U32"))
    print("MEAN:", {k: round(v, 4) for k, v in mean.items()},
          "SD:", {k: round(v, 4) for k, v in std.items()},
          "pooled_OOF_AUC", run["pooled_oof_auc"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
