#!/usr/bin/env python3
"""Compare paired directed and bidirectional runs using manifest-validated saved predictions."""
import argparse
import csv
from pathlib import Path
from _common import load_reports, validate_pairs, site_effects, infer, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--runs-a', required=True, type=Path)
    p.add_argument('--runs-b', required=True, type=Path)
    p.add_argument('--out-json', default='reproduced/r3/SYMMETRISED_VS_CANONICAL.json', type=Path)
    p.add_argument('--out-csv', default='reproduced/r3/SYMMETRISED_VS_CANONICAL.csv', type=Path)
    args = p.parse_args()
    effects = []
    for directory in (args.runs_a, args.runs_b):
        reports, expected = load_reports(args.manifest, directory)
        validate_pairs(reports, directory)
        effects.append((site_effects(reports)[0], len(expected)))
    a, b = effects[0][0], effects[1][0]
    out = dict(complete=True, status='parallel implementation-sensitivity analysis',
               canonical=dict(infer(a), fits=effects[0][1]),
               symmetrised=dict(infer(b), fits=effects[1][1]),
               sign_agreement=dict(agree=sum((a[s] > 0) == (b[s] > 0) for s in a), comparable_sites=len(a)))
    write_json(args.out_json, out)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['site', 'canonical_site_macro', 'symmetrised_site_macro', 'same_sign'])
        for site in sorted(a):
            writer.writerow([site, a[site], b[site], (a[site] > 0) == (b[site] > 0)])
    for name in ('canonical', 'symmetrised'):
        print(name, out[name]['population_macro'], out[name]['ci95'], 'sign p', out[name]['sign_p'])


if __name__ == '__main__':
    main()
