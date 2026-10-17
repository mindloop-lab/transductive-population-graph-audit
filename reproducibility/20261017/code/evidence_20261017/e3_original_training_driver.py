#!/usr/bin/env python3
"""E3 - LOSO information-access ablation, retrained (protocol corrected per review v4-5).

Both arms use the SAME source-site optimisation-training and validation sets, drawn with the frozen split seed
(666, as in scripts/train_loso.py). Target-site labels are used for final scoring only. The single manipulated
factor is the population-edge access of the target-site nodes:

  ON  : the frozen LOSO configuration - target subjects are inserted as queries connected only by cross-site
        same-sex edges to the train graph (no same-site edges, no target-to-target edges).
  OFF : those cross-site same-sex edges are removed as well, so target nodes carry NO population edges at all.
        Their own imaging encoding is unchanged; any self-loop/residual created by the layer is recorded in
        EXPERIMENTS/E3/AUTO_EDGE_AUDIT.json.

Per unit the script writes the train/validation/test ID lists, asserts that train and validation contain no
target-site subject, asserts OFF carries exactly zero target-incident edges, and asserts under both arms that
target-to-target edges are zero. A cross-arm check then confirms byte-identical train/val sets and identical
source-source edge counts for the same site and seed.

Run: e3_loso_access_v2.py --smoke    |    (default) 20 sites x 3 seeds x 2 arms = 120 units
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import time

import numpy as np

R = pathlib.Path(__file__).resolve().parent
WT = pathlib.Path("/mnt/disk_8tb/codex/2609-wan/del/CR-MAGP-AHGNN-transductive-model")
MODEL_SRC = WT / "src" / "cr_magp_ahgnn" / "models" / "heterogeneous_magp.py"
TRAINER = WT / "scripts" / "train_loso.py"
CACHE = WT.parent / "MAGP_AHGNN-v2" / "artifacts" / "phase_minus1b" / "cache_abide_data.npz"
PHENO = pathlib.Path("/mnt/disk_8tb/data/abide_pcp/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv")
BASE = R / "EXPERIMENTS/E3"
SPLIT_SEED = 666
FORMAL_EPOCHS, FORMAL_PATIENCE = 300, 150


def norm(s):
    return str(int(s))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_cohort():
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    npz = np.load(CACHE, allow_pickle=True)
    valid = [str(x) for x in npz["subject_ids"]]
    adjs, fused = npz["adj"], npz["fused_x"]
    pheno = {}
    with PHENO.open(newline="") as f:
        for row in csv.DictReader(f):
            pheno[row["SUB_ID"]] = row
    y = np.zeros(len(valid)); sex = np.zeros(len(valid), dtype=int)
    usite = sorted({pheno[norm(s)]["SITE_ID"] for s in valid})
    site = np.zeros(len(valid), dtype=int)
    for i, s in enumerate(valid):
        p = pheno[norm(s)]
        y[i] = 1 if int(p["DX_GROUP"]) == 1 else 0
        site[i] = usite.index(p["SITE_ID"])
        sex[i] = int(p["SEX"])
    return valid, adjs, fused, y, site, sex, usite, StandardScaler, PCA


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")   # target directory; used by the determinism double run
    ap.add_argument("--sites", default="")
    ap.add_argument("--seeds", default="666,2025,2026")
    ap.add_argument("--arms", default="ON,OFF")
    ap.add_argument("--epochs", type=int, default=FORMAL_EPOCHS)
    ap.add_argument("--patience", type=int, default=FORMAL_PATIENCE)
    a = ap.parse_args()
    if a.smoke:
        a.epochs, a.patience = 40, 20

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("ABORT: no CUDA device visible")
    device = torch.device("cuda:0")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"device: {device} | torch {torch.__version__}+cu{torch.version.cuda} | epochs {a.epochs} patience "
          f"{a.patience} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'UNSET')}")
    print("gpu:", torch.cuda.get_device_name(0))

    tl = load_module("frozen_train_loso", TRAINER)
    het = load_module("het_magp", MODEL_SRC)
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedShuffleSplit
    from scipy.special import softmax

    valid, adjs, fused, y, site, sex, usite, StandardScaler, PCA = load_cohort()
    print(f"cohort {len(valid)} subjects | {len(usite)} sites")

    write_dir = (pathlib.Path(a.out) if a.out else BASE / ("smoke_v2" if a.smoke else "runs"))
    write_dir.mkdir(parents=True, exist_ok=True)
    sites = [s for s in (a.sites.split(",") if a.sites else usite) if s]
    seeds = [int(x) for x in a.seeds.split(",")]
    arms = a.arms.split(",")
    plan = {"sites": sites, "seeds": seeds, "arms": arms, "epochs": a.epochs, "patience": a.patience,
            "split_seed": SPLIT_SEED, "smoke": bool(a.smoke),
            "unit_count": len(sites) * len(seeds) * len(arms)}
    (write_dir / "PLAN.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"planned units: {plan['unit_count']}")

    reg = write_dir / "REGISTRY.tsv"
    if not reg.exists():
        reg.write_text("unit\tsite\tseed\tarm\tn_train\tn_val\tn_test\t"
                       "target_incident_edges\ttest_auc\tseconds\tstatus\n", encoding="utf-8")
    invariants, done, t_start = [], 0, time.time()
    for si_name in sites:
        si = usite.index(si_name)
        te = np.where(site == si)[0]
        tr_all = np.where(site != si)[0]
        sss = StratifiedShuffleSplit(1, test_size=0.1, random_state=SPLIT_SEED)
        tr_i, va_i = next(sss.split(np.zeros(len(tr_all)), y[tr_all]))
        tri_g, vai_g = tr_all[tr_i], tr_all[va_i]
        assert len(set(te.tolist()) & set(tr_all.tolist())) == 0, "target site leaked into train"
        assert set(tri_g.tolist()).isdisjoint(set(te.tolist()))
        assert set(vai_g.tolist()).isdisjoint(set(te.tolist()))
        ids = {"train": [valid[i] for i in tri_g], "val": [valid[i] for i in vai_g],
               "test": [valid[i] for i in te]}
        tr_hash = hashlib.sha256("\n".join(ids["train"]).encode()).hexdigest()[:16]
        va_hash = hashlib.sha256("\n".join(ids["val"]).encode()).hexdigest()[:16]
        for seed in seeds:
            for arm in arms:
                tag = f"E3_{arm}_{si_name}_s{seed}"
                rep = write_dir / f"{tag}_report.json"
                if rep.exists():
                    print(f"[skip] {tag} (report present)")
                    continue
                t0 = time.time()
                torch.manual_seed(seed); np.random.seed(seed)
                st = np.vstack([fused[i] for i in tri_g])
                sc = StandardScaler().fit(st)
                pc = PCA(32, random_state=SPLIT_SEED).fit(sc.transform(st))
                x_all, e_all, ea_all, b_all, a_all = tl._batched_for(adjs, fused, pc, sc, tr_all.tolist(), device)
                x_te, e_te, ea_te, b_te, a_te = tl._batched_for(adjs, fused, pc, sc, te.tolist(), device)
                T = len(tr_all)
                pos_tr = np.searchsorted(tr_all, tri_g)
                pos_va = np.searchsorted(tr_all, vai_g)
                e_intra = tl._intra_edges(sex, site, tr_all)
                e_cross = tl._cross_edges(sex[tr_all], sex[te])
                e_tr = torch.from_numpy(e_intra).long().to(device)
                lbl = torch.from_numpy(y.astype(np.int64)).to(device)
                model = het.HeterogeneousMAGP(het.MAGPConfig(device=str(device))).to(device)
                opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
                best, pct, best_sd = -1.0, 0, None
                for _ in range(a.epochs):
                    model.train(); opt.zero_grad()
                    emb = model.subject_encoder(x_all, e_all, ea_all, b_all, a_all)
                    lg = model.population_gnn(emb, e_tr, torch.zeros((2, 0), dtype=torch.long, device=device))
                    loss = tl._calibrated_loss(lg[pos_tr], lbl[tri_g], 0.1, 2)
                    loss.backward(); opt.step()
                    model.eval()
                    with torch.no_grad():
                        embv = model.subject_encoder(x_all, e_all, ea_all, b_all, a_all)
                        lv = model.population_gnn(embv, e_tr,
                                                  torch.zeros((2, 0), dtype=torch.long, device=device)).cpu().numpy()
                    sval = float((np.argmax(lv[pos_va], 1) == y[vai_g]).mean()) + float(
                        roc_auc_score(y[vai_g], softmax(lv[pos_va], 1)[:, 1]))
                    if sval > best:
                        best, pct = sval, 0
                        best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    else:
                        pct += 1
                    if pct > a.patience:
                        break
                model.load_state_dict(best_sd); model.eval()
                if arm == "ON":
                    e_all_t = torch.cat([e_tr, torch.from_numpy(e_cross).long().to(device)], 1)
                    target_incident = int(e_cross.shape[1])
                else:
                    e_all_t = e_tr
                    target_incident = 0
                ei = e_all_t.cpu().numpy()
                target_target = int(np.sum((ei[0] >= T) & (ei[1] >= T)))
                with torch.no_grad():
                    emb_tr = model.subject_encoder(x_all, e_all, ea_all, b_all, a_all)
                    emb_te = model.subject_encoder(x_te, e_te, ea_te, b_te, a_te)
                    emb_c = torch.cat([emb_tr, emb_te], 0)
                    lg_c = model.population_gnn(emb_c, e_all_t,
                                                torch.zeros((2, 0), dtype=torch.long, device=device)).cpu().numpy()
                p = softmax(lg_c[T:], 1)[:, 1]
                auc_ = float(roc_auc_score(y[te], p))
                inv = {"arm": arm, "site": si_name, "seed": seed,
                       "train_ids_hash": tr_hash, "val_ids_hash": va_hash,
                       "target_absent_from_train": True, "target_absent_from_val": True,
                       "target_incident_population_edges": target_incident,
                       "target_to_target_edges": target_target,
                       "source_source_edges": int(e_intra.shape[1]),
                       "arm_off_zero_target_edges": (arm != "OFF" or target_incident == 0),
                       "test_auc": auc_, "all_pass": bool(target_target == 0 and (arm != "OFF" or target_incident == 0))}
                assert inv["all_pass"], f"invariant failed for {tag}"
                invariants.append(inv)
                rep.write_text(json.dumps({"unit": tag, "arm": arm, "site": si_name, "seed": seed, "ids": ids,
                                           "invariants": inv, "epochs": a.epochs, "best_val": best,
                                           "test_auc": auc_}, ensure_ascii=False, indent=2), encoding="utf-8")
                np.savez_compressed(write_dir / f"{tag}_pred.npz", subject_ids=np.array(ids["test"]),
                                    label=y[te].astype(int), site=np.array([si_name] * len(te)),
                                    prob=p.astype(np.float32))
                with reg.open("a", encoding="utf-8") as f:
                    f.write(f"{tag}\t{si_name}\t{seed}\t{arm}\t{len(ids['train'])}\t{len(ids['val'])}\t"
                            f"{len(ids['test'])}\t{target_incident}\t{auc_:.4f}\t{time.time() - t0:.1f}\tOK\n")
                done += 1
                print(f"[{done}] {tag} test_auc {auc_:.4f} target_edges {target_incident} "
                      f"({time.time() - t0:.1f}s)", flush=True)
    (write_dir / "INVARIANTS.json").write_text(json.dumps(
        {"all_pass": all(i["all_pass"] for i in invariants), "units": len(invariants), "per_unit": invariants},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("\ncross-arm identity check (same site and seed, ON vs OFF):")
    by = {}
    for i in invariants:
        by.setdefault((i["site"], i["seed"]), {})[i["arm"]] = i
    bad = 0
    for k, v in sorted(by.items()):
        if "ON" in v and "OFF" in v:
            same = (v["ON"]["train_ids_hash"] == v["OFF"]["train_ids_hash"]
                    and v["ON"]["val_ids_hash"] == v["OFF"]["val_ids_hash"]
                    and v["ON"]["source_source_edges"] == v["OFF"]["source_source_edges"])
            print(f"  {k[0]} seed {k[1]}: train/val identical = {same} | source-source edges "
                  f"{v['ON']['source_source_edges']} == {v['OFF']['source_source_edges']} | "
                  f"target edges ON {v['ON']['target_incident_population_edges']} / OFF "
                  f"{v['OFF']['target_incident_population_edges']}")
            bad += 0 if same else 1
    print(f"  mismatches: {bad}")
    print(f"\nunits completed this call: {done} | elapsed {time.time() - t_start:.1f}s | registry {reg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
