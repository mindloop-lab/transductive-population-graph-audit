# Transductive Population-Graph Audit

Reproducibility code and frozen results for:

> **Deconfounding transductive population-graph learning in multisite fMRI:**
> **tracing high cohort accuracy to same-site supervision**

The study audits a **cohort-visible transductive** evaluation regime: test-subject
nodes, imaging features and permitted phenotype/context relations may be present in the
population graph during optimization, while test labels are excluded from the loss.
Under that regime, the site-aware population graph reaches about 0.94-0.95 AUC on
ABIDE-I. Performance falls to about 0.52-0.53 AUC when the target site is held out,
and mechanism controls localise the cohort-visible gain to same-site supervision rather
than same-site topology alone.

## What is in here

| path | content |
|---|---|
| `src/cr_magp_ahgnn/` | clean reference model and supporting evaluation code |
| `scripts/` | training/control harnesses and release-integrity checks |
| `data_contract/` | recovered per-seed OOF generation partitions, reconstructed aggregation reference and public data contract |
| `results/result_registry_paperM_v4.csv` | **single numeric source** for reported release values |
| `results/e1_out/` | shipped per-subject OOF predictions for release-backed conditions, including two LOSO re-runs |
| `results/gim1_out/` | shipped baseline prediction artifacts |
| `results/subgroup_analysis_seed666.json` | subgroup/calibration derived statistics |
| `results/supplement_stats.json` | supplementary declared values |
| `docs/` | provenance, registry schema, reproduction guide and data statement |
| `environment/` | portable requirements plus the reference environment snapshot |

A few legacy scalar rows keep their original backing summaries unpublished in this
v1.0.0 artifact. They are marked `not redistributed`, and the validator rejects rows whose
backing path does not resolve.

## Quickstart (no ABIDE data required)

```bash
python -m pip install -r environment/requirements-ci.txt
bash run_smoke.sh
```

`run_smoke.sh` finishes with `SMOKE PASS`. It runs the LOSO code path on a synthetic
fixture; the output is a plumbing check, not a scientific result.

## Core reproduction

```bash
export ABIDE_CACHE=/path/to/cache_abide_data.npz
export ABIDE_PHENO=/path/to/Phenotypic_V1_0b_preprocessed1.csv
bash run_reproduce_core.sh
```

Optionally set `ABIDE_CACHE_NIAK` to run the NIAK unseen-site LOSO path as well.
`run_full.sh` is retained as a backward-compatible alias for the core harness.

The registry also covers archived and derived analyses. Shipped artifacts are validated
against a repository-relative provenance contract rather than by claiming that one command
retrains all 69 rows. See `docs/REPRODUCE.md` and `docs/REGISTRY_SCHEMA.md`.

## Release integrity

```bash
python scripts/validate_public_tree.py
bash scripts/scan_secrets.sh
python scripts/validate_registry.py
python scripts/validate_oof.py
python scripts/build_registry.py --check
sha256sum -c MANIFEST.sha256
```

GitHub Actions runs the public-tree, independent secret-scan, registry/provenance,
evidence round-trip, manifest and synthetic-smoke checks on pushes and pull requests.
The public release is built from the audited tree as a clean-root repository.

## Data

ABIDE-I is public and de-identified. **No primary imaging data are redistributed here**;
only derived predictions, frozen splits and statistics are included. See `docs/DATA.md`.

## Citation and licence

`CITATION.cff` describes the software artifact, released under **Apache License 2.0**
(`Apache-2.0`). The public release is built from the audited tree as a clean root; GitHub
Release `v1.0.0` will be archived with Zenodo and the DOI added to the citation metadata
and the article.

This repository is a reproducibility artifact, not a manuscript archive.
