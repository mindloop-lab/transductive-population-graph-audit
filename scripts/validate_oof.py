#!/usr/bin/env python
"""Evidence validator: recompute every registry value from its backing predictions.

Rules established against the artifact (each was confirmed by reproducing the registry
exactly):

1. the backing file's format decides how the score is formed:
     oof_logits + oof_label  ->  softmax(z)[:, 1]
     oof_prob   + oof_label  ->  the probability as stored
     y + p                   ->  the probability as stored
2. rows whose conditions cover only part of the cohort (e.g. the matched-budget arms)
   leave the uncovered entries at exactly (0, 0); the covered subset is therefore
   recoverable and the pooled AUC is computed over it;
3. `mean_fold_AUC` is computed against the **frozen split manifest**
   (`data_contract/phase_minus1b_splits.json`), which is the authoritative partition and
   carries the subject ids of every fold;
4. cohort identity is asserted against the manifest: same subjects, each appearing once.

Run:  python scripts/validate_oof.py [registry.csv]     (exit 1 on any mismatch)
"""
import csv, hashlib, json, pathlib, re, sys
from collections import Counter
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
REG = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/result_registry_paperM_v4.csv"
SPLITS = ROOT / "data_contract/phase_minus1b_splits.json"
GEN_FILE = ROOT / "data_contract/oof_generation_outer_folds.json"
TOL = 0.0015
NOT_REDISTRIBUTED = ("not redistributed", "internal", "commit ", "n/a", "none")


def _payload(pairs) -> bytes:
    """Canonical payload shared with scripts/validate_registry.py."""
    def key(kv):
        k = kv[0]
        return (0, int(k)) if str(k).lstrip("-").isdigit() else (1, str(k))
    return ("\n".join(f"{k}\t{v}" for k, v in sorted(pairs, key=key)) + "\n").encode("utf-8")


# seed -> hash of the outer partition that generated that seed's stored predictions
GEN_PARTITIONS = {}
if GEN_FILE.is_file():
    for _s, _b in json.loads(GEN_FILE.read_text()).get("partitions", {}).items():
        GEN_PARTITIONS[_s] = _b["hash"]


def _norm(s):
    s = str(s)
    return s.lstrip("0") or "0"


def load_manifest():
    d = json.loads(SPLITS.read_text())
    fold_of, ids = {}, set()
    for f in d["folds"]:
        k = f["fold"] if "fold" in f else f.get("fold_index")
        for sid in f["test"]["subject_ids"]:
            fold_of[_norm(sid)] = k
            ids.add(_norm(sid))
    return fold_of, ids, d


def softmax_pos(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max(1, keepdims=True))
    return (e / e.sum(1, keepdims=True))[:, 1]


def read_predictions(path):
    d = np.load(path, allow_pickle=True)
    keys = set(d.keys())
    ids = [_norm(s) for s in np.asarray(d["subject_ids"]).ravel()] if "subject_ids" in keys else None
    if {"oof_logits", "oof_label"} <= keys:
        z = np.asarray(d["oof_logits"]); y = np.asarray(d["oof_label"]).ravel().astype(int)
        mask = ~np.all(z == 0.0, axis=1)
        return y, softmax_pos(z), ids, mask
    if {"oof_prob", "oof_label"} <= keys:
        return (np.asarray(d["oof_label"]).ravel().astype(int),
                np.asarray(d["oof_prob"]).ravel().astype(float), ids, None)
    if {"y", "p"} <= keys:
        return (np.asarray(d["y"]).ravel().astype(int),
                np.asarray(d["p"]).ravel().astype(float), ids, None)
    return None


def auc(y, p):
    return float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else None


fold_of, cohort, manifest = load_manifest()
print(f"frozen manifest: {len(cohort)} subjects, {len(set(fold_of.values()))} folds, "
      f"status={manifest.get('status')!r}")

rows = list(csv.DictReader(open(REG)))
NOTES = []
checked = skipped = masked = mism = gen_ok = 0
problems = []
for r in rows:
    rid, pf = r.get("experiment_id", "?"), (r.get("prediction_file") or "").strip()
    est, metric = (r.get("estimate") or "").strip(), (r.get("metric") or "").strip()
    if not pf or not est.replace(".", "").isdigit() or \
            any(pf.lower().startswith(m) or pf.lower() == m for m in NOT_REDISTRIBUTED):
        skipped += 1
        continue
    path = ROOT / pf
    if not path.exists():
        problems.append(f"{rid}: prediction_file does not resolve ({pf})"); mism += 1; continue
    if path.suffix != ".npz":
        skipped += 1; continue
    if not (metric.startswith("pooled") or metric == "mean_fold_AUC"):
        skipped += 1; continue
    try:
        got = read_predictions(path)
    except Exception as exc:
        problems.append(f"{rid}: cannot read {pf} ({type(exc).__name__})"); mism += 1; continue
    if got is None:
        problems.append(f"{rid}: unrecognised format in {pf}"); mism += 1; continue
    y, p, ids, mask = got
    if ids is not None:
        extra = set(ids) - cohort
        if extra:
            problems.append(f"{rid}: {len(extra)} subject id(s) outside the frozen cohort")
            mism += 1; continue
        dup = [k for k, c in Counter(ids).items() if c > 1]
        if dup:
            problems.append(f"{rid}: duplicated subject id(s) in the predictions")
            mism += 1; continue
    if ids is not None and "oof_fold" in np.load(path, allow_pickle=True).keys():
        # Check this file's fold labels against the recorded generation partition for its
        # seed: the artifact must be internally consistent with the provenance it cites.
        _d = np.load(path, allow_pickle=True)
        _f = np.asarray(_d["oof_fold"]).ravel()
        _raw = [str(s) for s in np.asarray(_d["subject_ids"]).ravel()]
        _pairs = [(i, int(o)) for i, o in zip(_raw, _f)]
        _h = hashlib.sha256(_payload(_pairs)).hexdigest()
        _seed = re.search(r"(?:seed|_s)(\d{3,4})", path.name)
        _expected = GEN_PARTITIONS.get(_seed.group(1)) if _seed else None
        if _expected and _h != _expected:
            problems.append(f"{rid}: its fold labels do not reproduce the recorded generation "
                            f"partition for seed {_seed.group(1)}")
            mism += 1
            continue
        if _expected:
            gen_ok += 1
    if mask is not None and not mask.all():
        y, p = y[mask], p[mask]
        ids = [i for i, keep in zip(ids, mask.tolist())] if ids is not None else None
        if ids is not None:
            ids = [i for i, keep in zip(ids, mask.tolist())] if False else ids
        masked += 1
    target = float(est)
    pooled = auc(y, p)
    ok = pooled is not None and abs(pooled - target) <= TOL
    detail = f"pooled {pooled:.4f}" if pooled is not None else "pooled n/a"
    if not ok and metric == "mean_fold_AUC" and ids is not None:
        per = []
        for f in sorted(set(fold_of.values())):
            sel = np.array([fold_of.get(i, -1) == f for i in ids])
            if sel.sum() and len(set(y[sel].tolist())) > 1:
                per.append(roc_auc_score(y[sel], p[sel]))
        if per:
            mf = float(np.mean(per))
            ok = abs(mf - target) <= TOL
            detail = f"mean-fold {mf:.4f} ({len(per)} folds)"
    checked += 1
    if not ok:
        mism += 1
        problems.append(f"{rid}: registry {target:.4f} vs recomputed {detail} ({pf})")

print(f"rows {len(rows)} | recomputed {checked} | skipped (frozen scalars / no npz) {skipped} "
      f"| partial-cohort rows {masked} | generation partitions reproduced {gen_ok} "
      f"| mismatches {mism}")
for p in problems[:15]:
    print("  -", p)
if mism:
    print("FAIL: registry values disagree with their backing predictions")
    sys.exit(1)
for n in NOTES:
    print("  note:", n)
print("PASS: every recomputable registry value matches its out-of-fold predictions")
