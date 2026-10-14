# R3 recomputation contract

The versioned input archive is `results/r3/recompute_inputs.zip`. Its digest is in
`results/r3/RECOMPUTE_INPUTS.sha256` and the repository manifest. `unpack_inputs.py` checks the archive
and every member before extracting it. This is a derived-statistics release, not an end-to-end training release.

Each NPZ contains only the saved outer-test rows (`oof_role`, `oof_label`, `oof_logits`, `subject_ids`).
Each report retains identity, AUC, initialization hash and mask hash; operational environment and
training-curve fields have been removed. `INPUT_LINEAGE.csv` records the SHA256 of the original full
report and prediction file for every one of the 1,326 released fits. There is no model refit or logit modification.
The subset transformation changes file bytes; it does not claim byte identity with the originals.

Reduced manifests carry site, fold, feasibility and the frozen training seeds; they retain original
cell indices and omit training/validation subject lists. Thus they define all expected analysis units
without being enough to reconstruct the training design. P1 prefixes distinguish the three support draws.
Nine historical P1 reports have an unprefixed internal tag; the mapping is explicit in lineage and is
accepted only for those known prefix forms. Saved test labels and identities must match in each ON/OFF pair.
A missing whole cell, a missing arm/seed, an absent prediction file or identity mismatch aborts formal output.
Unexpected report files are skipped and counted. No empty directory can produce a FINAL result.

The shared inference reports equal-site averages, percentile site-resample intervals, absolute-tail
null-centred bootstrap tests with finite-simulation correction, and two-sided sign tests omitting exact
zero effects. These are conditional on the fitted models and seeds; shared training across sites limits
independence and exchangeability. The positive endpoint under directed edges does not survive bidirectional edges.
Per-site conditional intervals are secondary; withdrawn per-site tail-ratio tests and Holm values are not computed.

Default outputs go to `reproduced/r3/`, never to the committed records. Input directories, manifests and
outputs are command-line parameters. Use the README commands for population, comparison and support analyses.
For the conditional intervals:

```bash
python r3_analysis/major1_aggregate.py --manifest reproduced/r3_inputs/SPLIT_AND_MASK_MANIFEST.json --runs reproduced/r3_inputs/major1_runs --out-csv reproduced/r3/major1_site_level.csv --out-json reproduced/r3/major1_bootstrap.json
```

The full archived record will be assigned a new version DOI after release publication. Until then,
this repository commit and the input archive checksum identify the proposed R3 software version.
