"""Validated inputs and shared inference for the R3 statistical entry points."""
from pathlib import Path
import json
from math import comb
import numpy as np

SEEDS = [666, 2025, 2026]


def expected_units(manifest):
    m = json.loads(Path(manifest).read_text())
    seeds = m['seeds']['training']
    if seeds != SEEDS:
        raise ValueError('unexpected training seed set')
    prefix = m.get('run_prefix', '')
    expected = {}
    for i, cell in enumerate(m['cells']):
        if not cell['feasible']:
            continue
        for arm in ('ON', 'OFF'):
            for seed in seeds:
                tag = f"{prefix}cell{i:03d}_{cell['site']}_f{cell['fold']}_{arm}_s{seed}"
                expected[tag] = dict(cell=i, site=cell['site'], fold=cell['fold'], arm=arm, seed=seed)
    if not expected:
        raise ValueError('manifest has no official units')
    return expected


def load_reports(manifest, runs, *, allow_partial=False):
    expected = expected_units(manifest)
    runs = Path(runs)
    found = {p.name[:-len('_report.json')]: p for p in runs.glob('*_report.json')}
    missing = set(expected) - set(found)
    stray = set(found) - set(expected)
    if stray:
        print(f'[unit filter] skipped {len(stray)} non-manifest reports')
    if missing and not allow_partial:
        raise ValueError(f'missing {len(missing)} official runs: {sorted(missing)[:3]}')
    reports = []
    for tag in sorted(set(expected) & set(found)):
        report = json.loads(found[tag].read_text())
        if any(report.get(k) != v for k, v in expected[tag].items()):
            raise ValueError(f'report identity differs from manifest: {tag}')
        # Nine historical P1 reports had an unprefixed internal tag; file identity is explicit.
        if report.get('tag') not in (tag, tag.split('_', 1)[1] if tag.startswith(('r1_', 'r2_', 'r3_')) else tag):
            raise ValueError(f'invalid internal tag: {tag}')
        report = dict(report, tag=tag)
        load_predictions(runs, report)
        reports.append(report)
    return reports, expected


def auc(scores, labels):
    positive, negative = scores[labels == 1], scores[labels == 0]
    if not len(positive) or not len(negative):
        raise ValueError('test predictions must contain both classes')
    return float(((positive[:, None] > negative).sum() +
                  .5 * (positive[:, None] == negative).sum()) / (len(positive) * len(negative)))


def load_predictions(runs, report):
    path = Path(runs) / (report['tag'] + '_oof.npz')
    if not path.is_file():
        raise ValueError(f'missing prediction artifact: {path.name}')
    with np.load(path, allow_pickle=False) as archive:
        z = {k: archive[k] for k in ('oof_role', 'oof_label', 'oof_logits', 'subject_ids')}
    mask = z['oof_role'] == 'T'
    labels = z['oof_label'][mask]
    logits = z['oof_logits'][mask]
    ids = z['subject_ids'][mask]
    if not len(ids) or len(set(ids)) != len(ids) or not np.isin(labels, [0, 1]).all():
        raise ValueError(f'invalid test identities or labels: {report["tag"]}')
    if logits.shape != (len(labels), 2) or not np.isfinite(logits).all():
        raise ValueError(f'invalid logits: {report["tag"]}')
    exp = np.exp(logits.astype(float) - logits.max(axis=1, keepdims=True))
    score = (exp / exp.sum(axis=1, keepdims=True))[:, 1]
    if abs(auc(score, labels) - report['test_auc']) > 1e-9:
        raise ValueError(f'report AUC differs from saved predictions: {report["tag"]}')
    return z


def validate_pairs(reports, runs):
    pairs = {}
    for r in reports:
        key = (r['cell'], r['seed'])
        z = load_predictions(runs, r)
        mask = z['oof_role'] == 'T'
        signature = (z['subject_ids'][mask], z['oof_label'][mask])
        if key in pairs:
            if not all(np.array_equal(a, b) for a, b in zip(signature, pairs[key])):
                raise ValueError(f'ON/OFF subject or label mismatch: {key}')
        pairs[key] = signature


def site_effects(reports):
    by = {(r['cell'], r['arm'], r['seed']): r for r in reports}
    sites = sorted({r['site'] for r in reports})
    effects, per_seed = {}, {}
    for site in sites:
        cells = sorted({r['cell'] for r in reports if r['site'] == site})
        values = {}
        for seed in SEEDS:
            values[str(seed)] = float(np.mean([by[c, 'ON', seed]['test_auc'] -
                                              by[c, 'OFF', seed]['test_auc'] for c in cells]))
        effects[site] = float(np.mean(list(values.values())))
        per_seed[site] = values
    return effects, per_seed


def infer(effects, *, b=10000, seed=20261012):
    v = np.array([effects[s] for s in sorted(effects)], dtype=float)
    if not len(v) or not np.isfinite(v).all() or b <= 0:
        raise ValueError('invalid effect values or bootstrap size')
    pop = float(v.mean())
    idx = np.random.default_rng(seed).integers(0, len(v), size=(b, len(v)))
    ci = np.percentile(v[idx].mean(axis=1), [2.5, 97.5]).tolist()
    null = (v - pop)[idx].mean(axis=1)
    p = float((int((np.abs(null) >= abs(pop)).sum()) + 1) / (b + 1))
    nonzero = v[v != 0]  # exact zeros carry no sign information
    n = len(nonzero)
    positive = int((nonzero > 0).sum())
    tail = min(positive, n - positive)
    sign_p = min(1., 2 * sum(comb(n, i) for i in range(tail + 1)) / 2**n) if n else 1.
    return dict(n_sites=len(v), population_macro=pop, ci95=ci, null_centred_p=p,
                sign_k=positive, sign_n=n, sign_p=sign_p, per_site=effects)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
