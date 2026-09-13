# Provenance

## Release lineage

This repository is a curated reproducibility artifact assembled from a private research
and audit workspace, which retains the development, recovery and audit history.

The public artifact is created from the audited tracked tree as a **clean-root
repository**, so private commits, removed machine-local paths and audit-only Git identities
are not part of the public Git object database.

Some registry rows keep historical internal-origin identifiers, labelled as internal
references rather than publicly resolvable commits.

## Implementation

The shipped heterogeneous model is an independent clean reference implementation of the
recovered model core. Internal verification under strict checkpoint loading found outputs
agreeing to a maximum absolute difference of approximately 7e-7.

## Split provenance

Generation and aggregation use separate split records.

### OOF generation partitions

`data_contract/oof_generation_outer_folds.json` contains the outer-fold assignments
recovered from the redistributed OOF artifacts themselves. Separate partitions are stored
for seeds 666, 777, 2025 and 2026. The file states the canonical hash payload and records
that the inner validation split is not recoverable.

Generation SHA-256 values:

- seed 666: `5f6891d87155c1e33351a9e043925acd5672b86e04c50484f5c642d5e001afc6`
- seed 777: `a7b6d7327b2fe47092b99173e535e2c475bbc62e19f18a8289596a7eccb599ed`
- seed 2025: `d2a111bd6c63d37499afc7ad1a915051892fc208863a352a515ca8e266dff917`
- seed 2026: `c97a3a44c0f4f6053805fa31cac54a1fc4d94adb7ebcb1b366706ddd491cbd40`

### Reconstructed aggregation reference

`data_contract/phase_minus1b_splits.json` is the reconstructed protocol /
mean-fold aggregation reference. Its file SHA-256 is:

`45539f252a822f5267face1b52d7c3bb2c5fed6b90020442f3340aaff7d7cce4`

It is used only for reconstructed mean-fold aggregation.

## Determinism

PCA reductions use fixed `random_state` values and deterministic settings are enabled
where supported. For redistributed OOF artifacts, the recovered per-seed generation
partitions provide the relevant outer-fold provenance. The reconstructed aggregation
reference remains available for metrics that explicitly use it.

The reference software environment is recorded in `environment/versions.json`.

## Leave-one-site-out re-runs

Deterministic LOSO prediction dumps shipped in this repository include:

- C-PAC: `results/e1_out/loso_cpac/model_loso_predictions.npz`, pooled AUC 0.5216;
- NIAK: `results/e1_out/loso_niak/model_loso_predictions.npz`, pooled AUC 0.5317.

LOSO rows carry an explicit site-wise provenance status.

## Registry backing artifacts

Where a prediction/statistics artifact is redistributed, the registry points to its
repository-relative path. Rows without sufficient redistributed evidence carry an
explicit status such as `not_redistributed` or `not_recovered_outer_partition`.

## Public-tree boundary

Release CI validates the tracked tree with both the project-specific public-tree auditor
and an independent pinned Gitleaks scan. The clean-root public candidate repeats these
checks, and its complete Git history is scanned before publication.
