# v1.0 Release Audit

Status: **PRIVATE RELEASE CANDIDATE — FINAL GATES BEFORE CLEAN-ROOT**

## Scientific freeze

Release engineering leaves the frozen estimates, confidence intervals and prediction
arrays unchanged. The registry remains the single numeric source.

The release validators establish:

- 69 registry rows with zero scientific-field differences after provenance migration;
- four recovered outer generation partitions (seeds 666, 777, 2025 and 2026), each
  internally consistent across the artifacts that carry fold labels;
- 50/69 registry rows rebuilt from redistributed evidence and 19/69 explicitly carried
  with non-recomputed provenance status;
- zero field-level differences in the registry round-trip check.

## Split provenance

Two split records are kept separate.

| object | role |
|---|---|
| `data_contract/oof_generation_outer_folds.json` | outer partitions that generated stored OOF predictions, recovered from the artifacts themselves, one partition per seed |
| `data_contract/phase_minus1b_splits.json` | reconstructed protocol / mean-fold aggregation reference |

Generation-partition SHA-256 values use the canonical payload
`subject_id<TAB>fold<LF>`, with subject IDs sorted numerically:

- seed 666: `5f6891d87155c1e33351a9e043925acd5672b86e04c50484f5c642d5e001afc6`
- seed 777: `a7b6d7327b2fe47092b99173e535e2c475bbc62e19f18a8289596a7eccb599ed`
- seed 2025: `d2a111bd6c63d37499afc7ad1a915051892fc208863a352a515ca8e266dff917`
- seed 2026: `c97a3a44c0f4f6053805fa31cac54a1fc4d94adb7ebcb1b366706ddd491cbd40`

Only the outer fold assignment is recoverable; the inner validation split is not.

## Registry provenance

The registry schema uses explicit provenance fields:

- `generation_split_hash`: generation partition SHA-256 when the backing artifact
  supplies the fold labels, otherwise a non-recomputed status;
- `aggregation_split_hash`: aggregation-reference SHA-256 for rows whose reported value
  is aggregated with that reference;
- `prediction_file`: repository-relative backing artifact when redistributed;
- `code_commit`: origin/release provenance; private development hashes are not publicly
  resolvable commits.

`scripts/validate_registry.py`, `scripts/validate_oof.py` and
`scripts/build_registry.py --check` independently test schema/provenance, prediction
recomputation and field-level round-trip consistency.

## Public-tree hygiene

The tracked tree is the release surface; the private audit history is not.

The release removes machine-local one-off helpers, companion-paper/recovery material and
unused legacy code. Two independent checks are required:

1. `scripts/validate_public_tree.py` fails on machine-local paths, internal workspace
   identifiers, private-key headers and common credential/token shapes. AI/agent names are
   review-only signals rather than automatic failures.
2. `scripts/scan_secrets.sh` runs a pinned Gitleaks release against a `git archive` of
   the tracked tree, so secret scanning is independent of the custom validator.

The private repository retains the development and audit history; the public release
does not.

## Licence and publication authority

The release metadata consistently declares **Apache License 2.0** (`Apache-2.0`) in
`LICENSE`, `CITATION.cff` and `.zenodo.json`.

Technical licence metadata is therefore fixed. Making the repository public is a separate
decision and should occur only after the responsible rights holder has confirmed that the
software may be distributed under this licence.

## Release path

The authoritative publication path is:

```text
final audited private tree
  -> PUBLIC_TREE_FREEZE
  -> new clean-root repository (PRIVATE)
  -> full CI + full-history secret scan
  -> rights/publication approval
  -> PUBLIC
  -> GitHub Release v1.0.0
  -> Zenodo archive / DOI
  -> DOI backfill in README, CITATION.cff and the article
```

The audit pull request remains private and will not be merged into the public history.
It is kept as an audit trail.

## Gate to PUBLIC_TREE_FREEZE

Before creating the clean-root candidate, require all of the following on the final
private-tree HEAD:

- public-tree hygiene: PASS;
- independent secret scan: PASS;
- registry validator: PASS;
- OOF/evidence recomputation: PASS;
- registry round-trip: PASS;
- manifest verification: PASS;
- synthetic smoke: PASS;
- scientific numeric diff: 0.

The clean-root candidate repeats the same checks and additionally runs Gitleaks against
its complete history.
