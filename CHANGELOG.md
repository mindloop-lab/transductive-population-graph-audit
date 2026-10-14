# CHANGELOG

## R3 revision (2026-10-13/14) - corrections from two external review rounds

Withdrawn
  * tail-ratio bootstrap p values and any Holm adjustment computed from them (not calibrated tests).
  * per-site p values and per-site multiplicity correction (the across-site distribution is not a valid
    sampling unit for a single site).
  * the claim that the same propagation code underlies every comparison row.
  * the explanation that the same-site arm varies across support draws because the graph contains the
    varying other-site support: the graph is fixed across draws; the arm's supervised target set
    (C + S, C = O \ R) is what changes.

Added / corrected
  * paired supervision-location experiment with the population endpoint, site-resample interval and an
    exact sign test, implemented post hoc against a pre-specified endpoint (stated as such).
  * a full **symmetrised** parallel run (528 fits) showing the paired effect does not survive
    symmetrisation; reported beside the primary numbers, not instead of them.
  * pre-declared support-sampling sensitivity (270 fits) with a frozen decision rule.
  * graph semantics rewritten along the actually-loaded module path
    (`models/heterogeneous_magp.py`, dual-channel `TransformerConv`): no graph self-loops are added,
    a root-weight transform acts instead, and there is no row normalisation in this path.
  * statistical corrections: two-sided null tails defined by the absolute observed effect with the
    (b+1)/(B+1) finite-simulation correction; sign test uses the smaller tail (8/18 -> p = 0.815).
  * blanket "all comparisons significant after Holm" wording removed; comparisons whose archived inputs
    cannot be relocated are reported descriptively with no formal significance claim.
  * six missing `\input` caption dependencies resolved; both PDFs compile from a self-contained tree.
