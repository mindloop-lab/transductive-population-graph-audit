# Reproduction guide

## 1. Environment

For the lightweight release checks and synthetic smoke test:

```bash
python -m pip install -r environment/requirements-ci.txt
```

For the reference scientific environment:

```bash
python -m pip install -r environment/requirements.txt
python -c "import torch, torch_geometric, sklearn; print('ok')"
```

`environment/versions.json` records the reference package versions. The public
requirements contain no machine-local editable paths.

## 2. Release-integrity checks

```bash
python scripts/validate_public_tree.py
bash scripts/scan_secrets.sh
python scripts/validate_registry.py
python scripts/validate_oof.py
python scripts/build_registry.py --check
sha256sum -c MANIFEST.sha256
```

These gates separately check public-tree hygiene, secrets, the 21-column registry schema,
per-seed generation-split provenance, direct OOF evidence, the broader registry
round-trip and the frozen manifest. The reconstructed
`data_contract/phase_minus1b_splits.json` is an aggregation reference, not the
generation partition for the stored OOF predictions.

## 3. Smoke test (synthetic)

```bash
bash run_smoke.sh
```

This builds a 60-subject synthetic fixture and runs the LOSO harness. Expected last
line: `SMOKE PASS`. The fixture is only a software plumbing test.

## 4. Core real-data reproduction

```bash
export ABIDE_CACHE=<cpac_cache.npz>
export ABIDE_PHENO=<phenotype.csv>
# optional:
export ABIDE_CACHE_NIAK=<niak_cache.npz>
bash run_reproduce_core.sh
```

The C-PAC cache contract is:

- `subject_ids`: length N
- `fused_x`: (N, 116, 141)
- `adj`: (N, 116, 116)

The core harness executes the main cohort-visible model and the mechanism chain needed
to test the paper's central claim, including:

1. main transductive cohort model;
2. canonical sex+site graph;
3. site-only graph;
4. subject-only control;
5. random-graph control;
6. label-permutation null;
7. **train-test population-edge removal** via `--mask_tt`;
8. degree-preserving rewiring;
9. constant-feature topology control;
10. C-PAC unseen-site LOSO;
11. optional NIAK unseen-site LOSO.

## 5. Reference values

Rounded reference values for seed 666 include:

| quantity | value |
|---|---:|
| cohort pooled OOF AUC (sex+site) | ~0.941 |
| site-only edges | ~0.948 |
| subject-only (no graph) | ~0.587 |
| random graph | ~0.587 |
| label-permutation null | ~0.492 |
| no train-test population edges | ~0.562 |
| degree-preserving rewiring | ~0.597 |
| leave-one-site-out C-PAC | ~0.522 |
| leave-one-site-out NIAK | ~0.532 |

Exact estimates, intervals and provenance are frozen in
`results/result_registry_paperM_v4.csv`.

## 6. Scope of the frozen registry

The v1.0 registry has 69 rows. It includes the executable core mechanism chain plus
additional archived/derived analyses (head ablations, supervision masking, matched
budgets, baseline and summary statistics). The release does not claim that
`run_reproduce_core.sh` retrains all 69 rows.

When a backing artifact is shipped, `prediction_file` is a repository-relative path and
must exist. A small number of historical scalar-only rows are retained for manuscript
traceability but their original summary source was not redistributed; these rows are
explicitly labelled `not redistributed` and must not be mistaken for independently
recomputable OOF artifacts.

## 7. Determinism and split identity

The redistributed OOF artifacts carry recoverable outer-fold identity for four seed
groups; those generation partitions and their hashes are stored in
`data_contract/oof_generation_outer_folds.json`. The inner validation split is not
recoverable from the artifact and is not claimed.

The separate `phase_minus1b_splits.json` object is retained as the reconstructed
protocol / mean-fold aggregation reference.

PCA uses fixed `random_state` values and deterministic settings are enabled where
supported. Small floating-point differences may still occur across BLAS/CUDA stacks; the
frozen registry records the reference environment values.
