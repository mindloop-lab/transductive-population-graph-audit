#!/usr/bin/env python3
"""Population inference: pre-specified endpoint, analysis implemented post hoc.

Intervals condition on fitted predictions and fixed training seeds. Site resampling assumes
exchangeable site effects; shared training subjects limit that assumption. No per-site tests.
"""
import argparse
from pathlib import Path
import numpy as np
from _common import load_reports, validate_pairs, site_effects, infer, write_json, SEEDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--out-json', default='reproduced/r3/major1_population_inference.json', type=Path)
    args = parser.parse_args()
    reports, expected = load_reports(args.manifest, args.runs)
    validate_pairs(reports, args.runs)
    effects, seeds = site_effects(reports)
    result = infer(effects)
    result.update(status='post-hoc implementation of a pre-specified endpoint',
                  fits=len(expected), population_macro_per_seed={str(s): float(np.mean([v[str(s)] for v in seeds.values()])) for s in SEEDS},
                  inference_scope='fixed models and seeds; site-resample uncertainty; exchangeability is approximate',
                  per_site_inference='withheld: no valid per-site p or Holm values are computed',
                  withdrawn=['tail-ratio per-site p', 'Holm adjustments of withdrawn per-site p'])
    write_json(args.out_json, result)
    print('population_macro', result['population_macro'], 'CI', result['ci95'], 'sign p', result['sign_p'])


if __name__ == '__main__':
    main()
