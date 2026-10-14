# R3 analysis workspace — entry points

Order used for the reported results:

1. `r3_masks.py`            builds SPLIT_AND_MASK_MANIFEST.json (masks, eligibility, matching invariants)
1b `p1_masks.py`            support-sampling replicates (P1); same rules, only the draw seed changes
2. `major1_arms.py`         one fit: cell x arm x seed; `major1_arms_rev.py` and `major1_arms_sym.py` are the
                            reversed-orientation and symmetrised variants
3. `run_major1.py`          canonical campaign (528 fits); `run_p1_support.py` (270); `run_symmetrised.py` (528)
4. `major1_aggregate.py`    per-site aggregation with an integrity gate (recomputed AUC must equal the report)
5. `major1_population_inference.py`   the pre-specified population endpoint, implemented post hoc
6. `p1_aggregate.py`        support-sampling sensitivity and the ON-arm cross-replicate comparison
7. `sym_compare.py`         canonical vs symmetrised under the same inference procedure (refuses below 528/528)
8. `build_evidence_pack.py`, `package_r3_review.py`, `finalize_deliver_v2.sh`   packaging and delivery

Conventions: every script refuses to emit numbers from an incomplete run; no script overwrites another's
outputs; withdrawn p values (tail-ratio and per-site) are marked as such and never silently reused.
