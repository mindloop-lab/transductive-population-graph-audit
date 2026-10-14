#!/usr/bin/env python3
"""Aggregate three support draws. Graph is fixed; ON supervision C union S changes via C=O minus R.

ON is not a determinism control. Cross-draw differences are descriptive. The spread-based judgement
was not given a numerical cutoff before the runs; no causal location effect is inferred.
"""
import argparse
from pathlib import Path
from _common import load_reports, validate_pairs, site_effects, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifests', required=True, nargs=3, type=Path)
    p.add_argument('--runs', required=True, type=Path)
    p.add_argument('--out-json', default='reproduced/r3/P1_SUPPORT_SENSITIVITY.json', type=Path)
    args = p.parse_args()
    results, comparisons, total = {}, {}, 0
    for replicate, manifest in enumerate(args.manifests, 1):
        reports, expected = load_reports(manifest, args.runs)
        validate_pairs(reports, args.runs)
        total += len(expected)
        effects, seeds = site_effects(reports)
        for site in effects:
            results.setdefault(site, {})[str(replicate)] = dict(mean_over_seeds=effects[site], per_seed=seeds[site])
        for r in reports:
            if r['arm'] == 'ON':
                comparisons.setdefault((r['cell'], r['seed']), []).append(r['test_auc'])
    different = sum(len({round(x, 12) for x in values}) > 1 for values in comparisons.values())
    sites = {}
    for site, per_rep in results.items():
        values = [v['mean_over_seeds'] for v in per_rep.values()]
        sites[site] = dict(per_replicate=per_rep, across_replicate_mean=sum(values)/len(values),
                           across_replicate_spread=max(values)-min(values))
    write_json(args.out_json, dict(status='complete descriptive sensitivity analysis', fits=total, sites=sites,
               on_arm_cross_replicate=dict(pairs_compared=len(comparisons), differing_pairs=different,
               corrected_mechanism='fixed graph; ON supervised set C union S changes with support draw'),
               caveat='three pre-selected sites; no numeric threshold was pre-fixed; not a causal isolation'))
    print('fits', total, 'ON pairs', len(comparisons), 'differing', different)


if __name__ == '__main__':
    main()
