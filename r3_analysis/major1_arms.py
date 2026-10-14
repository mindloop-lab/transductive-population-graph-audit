#!/usr/bin/env python3
"""Major 1 — paired supervision-location arms driver (single cell x arm x seed).

Faithful reuse: the campaign's own `mechanism_controls.py` is imported as a module (its main() is
guarded by __main__), so the model, loss, feature batching and site-edge helpers are the campaign's
own code, not a re-implementation.

Design choices recorded here (arm-invariant by construction):
  * graph edges = same-site pairs among (S union O union T), i.e. every subject except the fixed
    validation set V -> identical in both arms, transductive for T, and V never contributes edges;
  * preprocessing (StandardScaler + PCA(32)) is fitted on the common pool S union O and only applied
    to V and T;
  * loss labels come from C union S (arm SAME_SITE_ON) or C union R (arm SAME_SITE_OFF) with equal
    total and per-class counts; V/T labels never enter the loss;
  * both arms share the training seed and are initialised identically immediately before model
    construction; optimiser state is separate and no arm is warm-started from the other.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
import time
from pathlib import Path

import numpy as np

PAPERM = Path("/mnt/disk_8tb/codex/2609-wan/paperM")
R3 = PAPERM / "PAPERM_CMIG_R3_20261011"
TREE = Path("/mnt/disk_8tb/codex/2609-wan/del/CR-MAGP-AHGNN-transductive-model")
CACHE = Path("/mnt/disk_8tb/codex/2609-wan/del/MAGP_AHGNN-v2/artifacts/phase_minus1b/cache_abide_data.npz")
PHENO = Path("/mnt/disk_8tb/data/abide_pcp/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv")


def load_module(name: str, path: Path):
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclasses resolve cls.__module__ through sys.modules
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=int, required=True, help="index into the mask manifest cells")
    ap.add_argument("--arm", required=True, choices=["ON", "OFF"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mm = load_module("mech", TREE / "scripts/mechanism_controls.py")
    import torch

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    het = load_module("het_magp", TREE / "src" / "cr_magp_ahgnn" / "models" / "heterogeneous_magp.py")

    man = json.loads((R3 / "SPLIT_AND_MASK_MANIFEST.json").read_text(encoding="utf-8"))
    cell = man["cells"][args.cell]
    assert cell["feasible"], "cell marked infeasible"
    npz = np.load(CACHE)
    ids = [str(x) for x in npz["subject_ids"]]; adjs = npz["adj"]; fused = npz["fused_x"]
    N = len(ids)
    pheno = {}
    with PHENO.open(newline="", encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            pheno[str(int(row["SUB_ID"]))] = row
    y = np.array([1 if int(pheno[str(int(s))]["DX_GROUP"]) == 1 else 0 for s in ids])
    site = np.array([pheno[str(int(s))]["SITE_ID"] for s in ids])
    usite = sorted(set(site.tolist()))
    site_id = np.array([usite.index(s) for s in site])
    sex = np.array([int(pheno[str(int(s))]["SEX"]) for s in ids])

    T = np.array(cell["T_test"]); S = np.array(cell["S_same_site_support"])
    V = np.array(cell["V_validation"]); O = np.array(cell["O_other_trainable"])
    R = np.array(cell["R_out_of_site_support"]); C = np.array(cell["C_common"])
    lab_idx = np.concatenate([C, S]) if args.arm == "ON" else np.concatenate([C, R])
    assert not (set(lab_idx.tolist()) & set(T.tolist())), "test index leaked into the loss"
    assert not (set(lab_idx.tolist()) & set(V.tolist())), "validation index leaked into the loss"
    if args.arm == "OFF":
        assert not (set(lab_idx.tolist()) & set(S.tolist())), "OFF arm must exclude target-site labels"
    pool = np.concatenate([S, O])

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    st = np.vstack([fused[i] for i in pool])
    sc = StandardScaler().fit(st); pc = PCA(32, random_state=args.seed).fit(sc.transform(st))
    x_all, e_all, ea_all, b_all, a_all = mm._batched_x(adjs, fused, pc, sc, "real", N=N, device=device)
    # arm-invariant graph: same-site edges among everyone except V
    edge_nodes = np.array(sorted(set(range(N)) - set(V.tolist())))
    by_site = {s: edge_nodes[site_id[edge_nodes] == s] for s in np.unique(site_id[edge_nodes])}
    pairs = []
    for s, nodes in by_site.items():
        if len(nodes) > 1:
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    pairs.append((int(nodes[i]), int(nodes[j])))
    enp = np.array(pairs, dtype=int).T
    e_t = torch.from_numpy(enp).long().to(device)

    torch.manual_seed(args.seed)  # identical initialisation in both arms
    model = het.HeterogeneousMAGP(het.MAGPConfig(device=str(device))).to(device)
    init_sha = hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes()
                                       for v in model.state_dict().values())).hexdigest()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
    labels = torch.from_numpy(y.astype(np.int64)).to(device)
    lab_t = torch.from_numpy(lab_idx.astype(np.int64)).to(device)

    best, pct, best_sd, best_ep, updates = -1.0, 0, None, -1, 0
    curve = []
    for ep in range(args.epochs):
        model.train(); opt.zero_grad()
        emb = model.subject_encoder(x_all, e_all, ea_all, b_all, a_all)
        lg = model.population_gnn(emb, e_t, torch.zeros((2, 0), dtype=torch.long, device=device))
        loss = mm._calibrated_loss(lg[lab_t], labels[lab_t], 0.1, 2)
        loss.backward(); opt.step(); updates += 1
        model.eval()
        with torch.no_grad():
            lv = model.population_gnn(model.subject_encoder(x_all, e_all, ea_all, b_all, a_all), e_t,
                                      torch.zeros((2, 0), dtype=torch.long, device=device)).cpu().numpy()
        from scipy.special import softmax
        from sklearn.metrics import roc_auc_score
        vl, vv = lv[V], y[V]
        val_auc = float(roc_auc_score(vv, softmax(vl, 1)[:, 1]))
        curve.append({"epoch": ep, "train_loss": float(loss.detach().cpu()), "val_auc": val_auc})
        if val_auc > best:
            best, best_ep, pct = val_auc, ep, 0
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            pct += 1
        if pct > args.patience:
            break

    model.load_state_dict(best_sd); model.eval()
    with torch.no_grad():
        lgT = model.population_gnn(model.subject_encoder(x_all, e_all, ea_all, b_all, a_all), e_t,
                                   torch.zeros((2, 0), dtype=torch.long, device=device)).cpu().numpy()
    from sklearn.metrics import roc_auc_score
    from scipy.special import softmax
    oof = np.zeros((N, 2)); oof[T] = lgT[T]
    yt = y[T]
    if len(np.unique(yt)) < 2:
        t_auc, auc_flag = None, "SINGLE_CLASS_TEST_FOLD"
    else:
        t_auc, auc_flag = float(roc_auc_score(yt, softmax(lgT[T], 1)[:, 1])), "OK"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tag = f"cell{args.cell:03d}_{cell['site']}_f{cell['fold']}_{args.arm}_s{args.seed}"
    rep = {"tag": tag, "cell": args.cell, "site": cell["site"], "fold": cell["fold"], "arm": args.arm,
           "seed": args.seed, "n": {k: len(v) for k, v in
                                    (("T", T), ("S", S), ("V", V), ("O", O), ("R", R), ("C", C))},
           "supervised": {"n": int(len(lab_idx)),
                          "classes": {str(k): int(v) for k, v in
                                      zip(*np.unique(y[lab_idx], return_counts=True))}},
           "test_auc": t_auc, "test_auc_flag": auc_flag, "best_val_auc": best, "best_epoch": best_ep, "updates": updates,
           "epochs_run": len(curve), "init_sha256": init_sha,
           "mask_sha256": cell["mask_sha256"], "curve": curve,
           "env": {"host": socket.gethostname(), "device": str(device),
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "torch": torch.__version__, "seed": args.seed}}
    (out / f"{tag}_report.json").write_text(json.dumps(rep, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out / f"{tag}_oof.npz", oof_logits=oof, oof_label=y,
                        oof_role=np.where(np.isin(np.arange(N), T), "T",
                                          np.where(np.isin(np.arange(N), V), "V",
                                                   np.where(np.isin(np.arange(N), S), "S",
                                                            np.where(np.isin(np.arange(N), R), "R",
                                                                     np.where(np.isin(np.arange(N), C), "C", ""))))),
                        subject_ids=np.array(ids, dtype="U32"))
    ta = "n/a" if t_auc is None else f"{t_auc:.4f}"
    print(f"[{tag}] test_auc {ta} ({auc_flag}) best_val {best:.4f} ep {best_ep} updates {updates} "
          f"init {init_sha[:12]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())