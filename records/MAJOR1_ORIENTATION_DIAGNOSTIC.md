# Orientation diagnostic (2026-10-13) — directed edges in the evaluated graph

Pre-fixed scope: cells 10-12 (KKI folds 0-2) x both arms x seed 666, identical code with only the edge
pair orientation reversed (`major1_arms_rev.py`), outputs in `major1_runs_orientation/`, never merged
into the canonical runs. Initial parameters identical within every pair (same seed), so a difference is
attributable to edge orientation alone.

| cell | arm | canonical | reversed | delta |
|---|---|---|---|---|
| 10 | OFF | 0.7333 | 0.6000 | -0.1333 |
| 10 | ON | 0.7333 | 0.7333 | +0.0000 |
| 11 | OFF | 0.2500 | 0.3333 | +0.0833 |
| 11 | ON | 0.2500 | 0.3333 | +0.0833 |
| 12 | OFF | 0.5000 | 0.7500 | +0.2500 |
| 12 | ON | 0.5000 | 0.3750 | -0.1250 |

n = 6 pairs; mean delta +0.0264; max |delta| 0.2500.

## Honest reading

* The check is **not** null: 5 of 6 pairs change, the largest change being 0.250 AUC. On 6-8 subject
  folds that magnitude is partly noise, but it is not negligible.
* The check is **underpowered**: 6 pairs, one seed. It neither establishes nor excludes a systematic
  orientation effect.
* Therefore the manuscript must state the code-level fact (directed `i<j` writes into a row-normalised
  matrix that also carries self-loops) together with this sensitivity, as a limitation. Orientation
  invariance must not be claimed. Symmetrising the graph and re-running the campaign is a material
  design change requiring separate, explicit authorisation; it is not done here.
* The same propagation code is shared with the Parisot reference row and with the reference model, so
  this limitation is a property of the harness rather than of one arm.

Status: internal diagnostic. Not a manuscript result; must not be cited as one.
