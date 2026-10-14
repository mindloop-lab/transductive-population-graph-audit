# Statistical family: 13 contrasts, enumerated (2026-10-12)

Source of the enumeration: the manuscript's own `tab:paired` caption -- "The archived analysis
scripts define 13 contrasts; this table presents the 12 within-protocol contrasts and retains their
Holm-adjusted values from the original 13-contrast calculation, which were not recomputed after
omission of the cross-protocol full-versus-supervision-masked contrast. That contrast, its interval
and its p-values are recorded in Online Resource 1, Supplementary Note S1."

## The 12 within-protocol contrasts (Table rows, verbatim order)

| # | contrast | dAUC | 95% CI | p | p_Holm | raw-p source found? |
|---|---|---|---|---|---|---|
| 1 | Full vs site-only edges | -0.007 | [-0.028, 0.014] | 0.520 | 0.520 | YES (full_sex_site vs full_site) |
| 2 | Full vs sex-only edges | +0.376 | [0.321, 0.418] | <0.001 | 0.009 | YES |
| 3 | Full vs random edges | +0.353 | [0.300, 0.407] | <0.001 | 0.009 | YES |
| 4 | Full vs subject-only (no graph) | +0.353 | [0.286, 0.400] | <0.001 | 0.009 | **no** |
| 5 | Full vs within-site shuffle (B) | -0.020 | [-0.048, -0.000] | 0.049 | 0.099 | **no** |
| 6 | Full vs no train-test edges (C) | +0.379 | [0.325, 0.426] | <0.001 | 0.009 | **no** |
| 7 | Site-only vs no train-test edges (C) | +0.386 | [0.337, 0.428] | <0.001 | 0.009 | **no** |
| 8 | Site-only vs degree-preserving rewiring (D) | +0.351 | [0.300, 0.402] | <0.001 | 0.009 | **no** |
| 9 | Dual- vs single-channel head | +0.394 | [0.337, 0.453] | <0.001 | 0.009 | YES |
| 10 | Dual-channel vs GAT head | +0.432 | [0.374, 0.479] | <0.001 | 0.009 | YES |
| 11 | Dual-channel vs GCN head | +0.450 | [0.382, 0.492] | <0.001 | 0.009 | YES |
| 12 | Dual-channel vs MLP head | +0.369 | [0.316, 0.414] | <0.001 | 0.009 | YES |

13th (cross-protocol, Online Resource 1 / Supplementary Note S1): full vs supervision-masked.
raw-p source found? **no** within the tables located on this host.

## Coverage verdict

* raw p values are locatable for **7 of the 13** items (rows 1-3 and 9-12) and **not** for 6
  (rows 4-8 and the cross-protocol item). The raw-p table that exists
  (`paperM/phase2_statistics/statistics/paired_effects.csv`, 10 rows) contains the *same* contrasts
  as the lineage map and **none** of the missing six.
* Therefore **Holm recomputation remains blocked for 6 of 13**. This does not by itself contradict
  the manuscript, whose caption already discloses that the Holm values were carried over and not
  recomputed after omitting the cross-protocol contrast.
* No p value may be inserted, re-derived from a rounded CI, or back-computed. If the reviewer
  requires recomputation, the missing six must be regenerated from retained predictions with a
  named script and digest, and the family size (13) held fixed.

## Path-base wrinkle (must be stated in any evidence index)

The lineage map records the raw-p source as the relative path `statistics/paired_effects.csv`. That
path resolves against the **working bundle** `paperM/phase2_statistics/`, which is **not** part of
`released_repo`. Any consolidated R3 evidence pack must either include that file or rebase the
reference explicitly; a bare relative path with no base tree is not re-checkable.
