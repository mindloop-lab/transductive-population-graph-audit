"""Validation-only Phase 2B incrementality closure decision."""

from __future__ import annotations


def closure_decision(
    *,
    reference_auc: float,
    pearson: float,
    b1_wrong_reference_right: int,
    b1_right_reference_wrong: int,
    max_success_site_share: float,
) -> str:
    net = b1_wrong_reference_right - b1_right_reference_wrong
    if reference_auc < 0.55 and net <= 0:
        return "REFERENCE_UNINFORMATIVE_STOP"
    if net >= 1 and max_success_site_share <= 0.5:
        return "REFERENCE_COMPLEMENTARY_FUSION_PILOT_ALLOWED"
    if abs(pearson) >= 0.8 or net <= 0 or max_success_site_share > 0.5:
        return "REFERENCE_REDUNDANT_STOP"
    return "REFERENCE_UNINFORMATIVE_STOP"
