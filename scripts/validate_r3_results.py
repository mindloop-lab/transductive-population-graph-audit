#!/usr/bin/env python3
"""Validate all 1,326 compact prediction inputs and independently rebuild the R3 endpoints."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)


with tempfile.TemporaryDirectory() as td:
    td=Path(td);inputs=td/'inputs';output=td/'results'
    run(ROOT/'r3_analysis/unpack_inputs.py','--output',inputs)
    run(ROOT/'r3_analysis/sym_compare.py','--manifest',inputs/'SPLIT_AND_MASK_MANIFEST.json',
        '--runs-a',inputs/'major1_runs','--runs-b',inputs/'major1_runs_sym',
        '--out-json',output/'comparison.json','--out-csv',output/'comparison.csv')
    actual=json.loads((output/'comparison.json').read_text())
    expected=json.loads((ROOT/'records/SYMMETRISED_VS_CANONICAL.json').read_text())
    for family in ['canonical','symmetrised']:
        for key in ['population_macro','null_centred_p','sign_p']:
            if abs(actual[family][key]-expected[family][key])>1e-12:
                raise SystemExit(f'R3 mismatch: {family}/{key}')
        if any(abs(a-b)>1e-12 for a,b in zip(actual[family]['ci95'],expected[family]['ci95'])):
            raise SystemExit(f'R3 interval mismatch: {family}')
    run(ROOT/'r3_analysis/major1_population_inference.py','--manifest',inputs/'SPLIT_AND_MASK_MANIFEST.json',
        '--runs',inputs/'major1_runs','--out-json',output/'population.json')
    population=json.loads((output/'population.json').read_text())
    if abs(population['population_macro']-actual['canonical']['population_macro'])>1e-12:
        raise SystemExit('population and comparison entry points disagree')
    run(ROOT/'r3_analysis/p1_aggregate.py','--manifests',*[inputs/f'P1_SUPPORT_MANIFEST_r{i}.json' for i in [1,2,3]],
        '--runs',inputs/'P1_support_runs','--out-json',output/'support.json')
    support=json.loads((output/'support.json').read_text())
    if support['fits']!=270 or support['on_arm_cross_replicate']['differing_pairs']!=34:
        raise SystemExit('support sensitivity identity/count mismatch')
    values={'LEUVEN_1':[.0648148148148148,-.07222222222222223,-.0240740740740741],
            'PITT':[.07111111111111114,.11666666666666668,.0068888888888889],
            'YALE':[.6516666666666667,.653611111111111,.5722222222222223]}
    for site,expected_values in values.items():
        for i,value in enumerate(expected_values,1):
            if abs(support['sites'][site]['per_replicate'][str(i)]['mean_over_seeds']-value)>1e-12:
                raise SystemExit(f'support effect mismatch: {site}/{i}')
print('PASS: 1,326 saved-prediction AUCs, population effects, intervals, corrected tests and support effects')
