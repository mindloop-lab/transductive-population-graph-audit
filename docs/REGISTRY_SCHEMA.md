# Registry schema

`results/result_registry_paperM_v4.csv` contains one row per frozen reported
experiment/statistic with **21 columns**:

`experiment_id, model, pipeline, protocol, graph_setting, mechanism_intervention, seed,
metric, score_definition, estimate, ci_low, ci_high, ci_subj_low, ci_subj_high, n,
n_positive, prediction_file, generation_split_hash, aggregation_split_hash, code_commit,
note`

## Release contract

`scripts/validate_registry.py` enforces:

1. the exact 21-column schema;
2. numeric metric/interval fields are in [0, 1] when present;
3. `ci_low <= estimate <= ci_high`;
4. `n` and `n_positive` are integer-valued and `n_positive <= n`;
5. text columns cannot silently parse as numeric column-shift artifacts;
6. `metric` is non-empty;
7. seed values encoded in experiment identifiers agree with the `seed` column;
8. non-empty `prediction_file` values are repository-relative paths and resolve;
9. missing/non-recomputed backing evidence is represented explicitly rather than by a
   fabricated path or split hash;
10. `generation_split_hash` is either a verified recovered generation-partition SHA-256
    or an allowed explicit provenance status;
11. `aggregation_split_hash` is populated only when the reported statistic explicitly
    uses the reconstructed aggregation reference;
12. `code_commit`/origin provenance is non-empty;
13. machine-local absolute paths are rejected;
14. the four recovered generation partitions and the aggregation reference are recomputed
    from shipped evidence rather than trusted as opaque strings.

Despite its historical name, `prediction_file` is the row's public backing-artifact
path and may point to an NPZ or JSON statistics artifact.

## Split provenance

Two distinct split objects are represented.

| object | meaning |
|---|---|
| `data_contract/oof_generation_outer_folds.json` | outer partitions that generated stored OOF predictions, recovered from the artifacts, one partition per seed |
| `data_contract/phase_minus1b_splits.json` | reconstructed protocol / mean-fold aggregation reference |

The generation file defines a canonical payload of
`subject_id<TAB>fold<LF>` sorted by numeric subject ID. The inner validation split is
not recoverable from the redistributed artifacts and is not claimed.

The aggregation reference has file SHA-256
`45539f252a822f5267face1b52d7c3bb2c5fed6b90020442f3340aaff7d7cce4`,
but it is not presented as the source partition for stored OOF predictions.

## Recomputing the values

`scripts/validate_oof.py` re-derives directly supported OOF metrics from redistributed
prediction files and checks cohort/fold provenance where available.

`scripts/build_registry.py --check` performs the broader registry round trip. It
currently reconstructs all metrics supported by the shipped evidence, including pooled
and mean-fold AUC, site-macro/site-median AUC, Brier score and ECE, while explicitly
carrying rows that cannot be independently rebuilt from redistributed evidence.

The builder never maintains a second long-lived numeric table and does not hard-code
scientific result values. It compares a temporary reconstruction field by field against
the canonical registry.

## Updating results

The v1.0 registry is a frozen release artifact. Do not edit frozen scientific estimates
in place. A future scientific update should create a new registry version, regenerate its
backing artifacts, update the manifest and issue a new release/DOI.
