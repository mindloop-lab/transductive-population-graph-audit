# P1 support-sampling sensitivity — final report (2026-10-14, corrected)

Frozen design: three pre-declared sites; support draws r=1,2,3 with pre-fixed seeds 20261014/15/16;
training seeds {666,2025,2026}; only the support draw changes (T/S/V identical to the canonical masks;
R and its complement C change by construction). 270 fits, separate directory and ledger.

## Frozen decision rule
If the across-replicate spread of the site effect is comparable to the effect itself, supervision
location and support composition are not separable at this sample size and the manuscript must say so.

| site | per-replicate site macro | mean | spread | rule outcome |
|---|---|---|---|---|
| LEUVEN_1 | r1:+0.065 / r2:-0.072 / r3:-0.024 | -0.010 | 0.137 | NOT separable |
| PITT | r1:+0.071 / r2:+0.117 / r3:+0.007 | +0.065 | 0.110 | NOT separable |
| YALE | r1:+0.652 / r2:+0.654 / r3:+0.572 | +0.626 | 0.081 | location effect exceeds the draw spread; composition still changes with it |

## Cross-replicate comparison of the same-site arm (corrected)

The ON arm differs across draws in **34 of 45** comparable
(cell, seed) pairs. **Corrected mechanism**: the population graph is *fixed* across draws (all nodes
except V); the ON arm's supervised target set is C ∪ S with C = O \\ R, so its **label composition**
changes with the draw. The earlier explanation — that the graph contains the varying other-site support —
was wrong and is retracted, and the ON arm is **not** a valid determinism control.

## Caveats
* Three sites only, pre-declared for support-size spread; not a population estimate.
* This is a sensitivity analysis and is never merged with, or substituted for, the Major 1 population endpoint.
* The rule's threshold was not numerically pre-fixed; the ratio is reported so the judgement is auditable.
