"""Explicitly separated Track R and Track S protocol registry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReproductionProtocol:
    public_name: str
    track: str
    implementation: str
    legacy_core_protocol: str | None
    query_semantics: str
    historical_acc: float | None
    historical_auc: float | None
    authoritative_for_training: bool


REPRODUCTION_PROTOCOLS = {
    "legacy_full_transductive": ReproductionProtocol(
        "legacy_full_transductive", "R", "recovered_protocol_core", "full_transductive",
        "all_subjects_joint_population_graph", 0.9013, 0.9646, True,
    ),
    "legacy_semi_transductive": ReproductionProtocol(
        "legacy_semi_transductive", "R", "recovered_protocol_core", "semi_transductive",
        "joint_graph_without_test_test_edges", 0.9001, 0.9602, True,
    ),
    "legacy_inductive_ref_batch": ReproductionProtocol(
        "legacy_inductive_ref_batch", "R", "recovered_protocol_core", "inductive_ref",
        "historical_joint_query_batch", 0.8450, 0.9360, True,
    ),
    "strict_single_query": ReproductionProtocol(
        "strict_single_query", "S", "clean_m0s", None,
        "frozen_reference_one_isolated_query", None, None, False,
    ),
}


def get_reproduction_protocol(name: str) -> ReproductionProtocol:
    try:
        return REPRODUCTION_PROTOCOLS[name]
    except KeyError as error:
        raise KeyError(f"unknown dual-track protocol: {name}") from error
