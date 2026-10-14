> Historical snapshot retained for traceability. It is not the current release contract.
> Per-site tail-ratio p values/Holm results and the former graph-change explanation are withdrawn.
> Current input checks and executable results are documented in docs/R3_RECOMPUTE.md.

# P1 interim finding: the "same-site" arm is not invariant to the support draw (2026-10-13)

Before running, the ON arm was expected to be invariant across the three support-sampling replicates,
since its supervised support is drawn from the target site. **The data refute that expectation.**

* across the 21 (cell, seed) pairs with two or more replicates available so far, the ON arm differs in
  **15** and the OFF arm differs in **14**;
* CORRECTED mechanism: the graph is fixed across draws (all nodes except V); what changes is the ON arm's
  supervised target set C ∪ S with C = O \\ R, so its label composition changes with the draw. The earlier
  graph-based explanation is retracted.

Consequences, stated before the campaign finishes:

1. The comparison is not a clean isolation of supervision location even in its own design: location and
   the composition of the support subjects remain entangled through the graph as well as through the
   supervised set.
2. The frozen decision rule's first branch therefore applies: if the across-replicate spread is
   comparable to the effect, the manuscript must say location and composition are not separable at this
   sample size, and no causal wording is licensed.
3. The earlier "determinism control" label in the aggregation script was wrong and has been replaced by
   "graph-mediated cross-replicate variability"; the numbers were not smoothed or dropped.
