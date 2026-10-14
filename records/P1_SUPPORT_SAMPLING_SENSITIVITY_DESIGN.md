# P1 frozen design — support-sampling composition sensitivity

Status: **design frozen, NOT yet run** (running it is a separate, explicit step: it needs new fits).

Reason it exists: the review's point is correct — currently there is exactly one fixed cross-site
support sampling per cell, so *supervision location* and *the composition of the support subjects*
are entangled. This design varies composition while holding the paired-protocol machinery fixed.

## Frozen choices (fixed before any run)

* Sites: three **pre-declared** sites spanning support sizes (small / medium / large), named here
  before running: LEUVEN_1, PITT, YALE. Chosen for support-set size spread, not for effect size, and
  not revisable after seeing results.
* Arms: SAME_SITE_ON and SAME_SITE_OFF, identical to Major 1 (same driver, same masks, same
  matching rule: equal supervised total and equal class counts).
* Seeds: {666, 2025, 2026} (the same three; no new seeds).
* Extra factor: **support sampling replicate r in {1,2,3}** for the OFF arm's other-site support draw
  (the ON arm is fully determined by the target site). Replicate r uses a pre-fixed seed
  `20261013 + r` so the draws are reproducible and cannot be selected on outcome.
* Endpoint: the same population quantity as Major 1 (equal-weight site mean of the within-site fold
  macro), reported **per replicate** and as the spread across replicates -- explicitly a
  sensitivity analysis, not a new primary result.
* Budget: 3 sites x 5 folds x 2 arms x 3 seeds x 3 replicates = 270 fits, estimated 4-6 h at the
  measured ~65 s/fit. Resumable, separate output directory, never merged into `major1_runs/`.
* Decision rule fixed in advance: if the across-replicate spread of the site effect is comparable to
  the effect itself, the manuscript must say that supervision location and support composition are not
  separable at this sample size; if the spread is small, the entanglement limitation can be narrowed
  in wording but the causal wording is still not licensed.
