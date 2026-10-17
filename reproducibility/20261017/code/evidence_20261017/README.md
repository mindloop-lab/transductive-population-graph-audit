# Retained supplementary evidence, 17 October 2026

This folder supports numerical recomputation, not complete historical training reconstruction.

Run from the manuscript root:

```bash
python3 -m pip install -r evidence_20261017/requirements.txt
python3 evidence_20261017/recompute.py --controls
python3 figure_sources/make_loso_access.py
```

Outputs go to `evidence_20261017/recomputed/`; delivered inputs are not overwritten.
The full recomputation checks the empirical-prior scores and interval, main-model
softmax AUC, paired-prior count arithmetic, E3 site-macro contrast and interval,
and all nine separate control-run contrasts and intervals. Original paired-prior
supervision sets are documented by `split_masks/SPLIT_AND_MASK_MANIFEST.json`.

## Input identities

- `split_masks/seed666_fast_oof.npz`: supplied main-model run, AUC 0.939608916.
- `site_prior_main/`: reconstructed ten-fold empirical site-prior rule, AUC 0.505622892.
- `control_predictions/`: separate control campaign; Full AUC 0.949841997.
- `E3_predictions.csv`: only E3 rows from the uploaded true-state review package;
  no E1 or E2 predictions are adopted as scientific evidence here.
- `E3_seed_endpoints.csv`, `E3_site_endpoints.csv`, `E3_recomputed_summary.json`:
  independent summaries of those predictions.
- `e3_original_training_driver.py`: unchanged supplied driver. It requires external
  model, helper and feature-cache inputs and is not a portable training entry point.
  Its arm switch occurs at scoring after source-only training.

The source review ZIP SHA256 is
`d3d3f754025056e9857472c1456df6d8012c610504a5ffade579f47c1b6951bd`.
No historical headline run is identified merely by nearby AUC values. No old
p-value/Holm family is endorsed by these new intervals. Seeds are fixed repeated
fits, not independent replicated subjects. Intervals condition on saved predictions.

E1 protocol-mismatched diagnostics and E2 split-deviating added-access experiments
are excluded. They are described in Supplementary Note S9; this folder does not
repackage them as accepted tests. No further model training was performed for this
manuscript revision.

These new inputs have not been published to GitHub, Zenodo or bioRxiv. The legacy
DOI and commit in the manuscript identify only their explicitly stated scopes.
