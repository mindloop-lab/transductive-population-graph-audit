# Transductive Population-Graph Audit — R3 release candidate

Code and derived evidence for **Reliable Evaluation of Transductive Population Graph Neural Networks
for Multisite Autism fMRI**. The proposed software version is **1.2.0**; it is not published yet.

## Results and interpretation

- Cohort out-of-fold AUC is about 0.94; unseen-site AUC is 0.522 (C-PAC) / 0.532 (NIAK).
- Under the directed paired protocol, the equal-site mean ON−OFF AUC effect is +0.252
  [0.184, 0.322], with 17/18 sites positive.
- With bidirectional edges the same endpoint is −0.003 [−0.020, 0.015], with 8/18 sites positive.
  The positive paired effect is implementation-conditional, not a robust causal supervision-location effect.
- Support draws change the ON supervised subject set, not the graph. The three-site sensitivity is descriptive.

## Three reproducibility levels

1. **Runnable here:** original release checks and synthetic smoke; R3 primary endpoints and comparison
   recomputation from the compact saved test predictions in `results/r3/recompute_inputs.zip`.
2. **Additional historical evidence:** complete training curves, operational manifests and the original
   feature caches are not bundled. The compact bundle records original file hashes in `INPUT_LINEAGE.csv`.
3. **Training reconstruction:** original core reproduction accepts user-provided ABIDE caches;
   end-to-end rebuilding of R3 training inputs from raw ABIDE is not supplied.

The R3 bundle contains test rows only from 528 directed, 528 bidirectional and 270 support fits.
Labels, subject identifiers and logits are unchanged. Training curves and environment fields are omitted.
Nine historical P1 tags are mapped to their prefixed filenames; one CALTECH smoke fit is excluded.
No primary imaging data are redistributed. See `docs/R3_RECOMPUTE.md` for lineage and limits.

## Quickstart

```bash
python -m pip install numpy scikit-learn pyyaml
python r3_analysis/unpack_inputs.py
python r3_analysis/major1_population_inference.py --manifest reproduced/r3_inputs/SPLIT_AND_MASK_MANIFEST.json --runs reproduced/r3_inputs/major1_runs
python r3_analysis/sym_compare.py --manifest reproduced/r3_inputs/SPLIT_AND_MASK_MANIFEST.json --runs-a reproduced/r3_inputs/major1_runs --runs-b reproduced/r3_inputs/major1_runs_sym
python r3_analysis/p1_aggregate.py --manifests reproduced/r3_inputs/P1_SUPPORT_MANIFEST_r1.json reproduced/r3_inputs/P1_SUPPORT_MANIFEST_r2.json reproduced/r3_inputs/P1_SUPPORT_MANIFEST_r3.json --runs reproduced/r3_inputs/P1_support_runs
```

For conditional within-site intervals run `major1_aggregate.py` with `--manifest` and `--runs`.
It computes no per-site p values or Holm adjustments. These intervals condition on fixed fitted models
and seeds. All R3 tools verify the full expected run set and saved-prediction AUCs before formal output.

## Original model paths and release checks

```bash
python -m pip install -r environment/requirements-ci.txt
bash run_smoke.sh
python scripts/validate_public_tree.py
python scripts/validate_metadata.py
python scripts/validate_registry.py
python scripts/validate_oof.py
python scripts/build_registry.py --check
python -m unittest discover -s tests
sha256sum -c MANIFEST.sha256
```

Core retraining: set `ABIDE_CACHE` and `ABIDE_PHENO` and run `bash run_reproduce_core.sh`.
See `docs/REPRODUCE.md` and `docs/DATA.md`. Smoke output is a plumbing check, not a scientific result.
Historical evidence limitations and withdrawn analyses are retained in `records/`; machine-specific
paths in those notes have been redacted. They are not executable input contracts.

## Citation and archive

Software is Apache-2.0. Project concept DOI: https://doi.org/10.5281/zenodo.22748743.
Version 1.1.0 DOI: https://doi.org/10.5281/zenodo.22892320 (does not include R3 additions).
The proposed 1.2.0 version DOI has not been minted. `CITATION.cff` currently uses the concept DOI;
replace the version DOI field only after the new archive is published. The preprint is
https://doi.org/10.64898/2026.09.15.751719. Publishing a release may trigger the configured Zenodo integration.
