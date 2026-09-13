"""H1: leak-free, site-free sex-relation population edges.

Paper M / MedIA gate H1 builds the population graph without any SITE_ID
signal. Only two explicit relation types are used, split by sex equality:

    R_same  = (u, v) with sex[u] == sex[v]
    R_diff  = (u, v) with sex[u] != sex[v]

Edge convention (must match the repo edge contract): ``edge_index[0]`` is the
SOURCE, ``edge_index[1]`` is the TARGET. Query attachment is strictly one-way
(source = reference, target = query); there are never query--query edges and
the reference subgraph is untouched by query insertion.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = [
    "reference_relations",
    "query_attachment",
]


def _as_int_array(sex) -> np.ndarray:
    return np.asarray(sex, dtype=np.int64).reshape(-1)


def _upper_triangle_edges(n: int) -> np.ndarray:
    """All (i, j) with i < j, stored as an (2, E) source/target array."""
    src = []
    dst = []
    for i in range(n):
        for j in range(i + 1, n):
            src.append(i)
            dst.append(j)
    if not src:
        return np.zeros((2, 0), dtype=np.int64)
    return np.stack((np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)))


def reference_relations(sex: "Sequence | np.ndarray") -> dict[str, np.ndarray]:
    """Split all reference--reference pairs (upper triangle, i<j) by sex.

    Returns ``{"same": (2, E_s) int array, "diff": (2, E_d) int array}`` where
    row 0 is source and row 1 is target. Deterministic; independent of input
    permutation up to the identity relabelling.
    """
    sex_arr = _as_int_array(sex)
    n = sex_arr.shape[0]
    edges = _upper_triangle_edges(n)
    if edges.shape[1] == 0:
        return {"same": edges, "diff": edges}
    u = edges[0]
    v = edges[1]
    same_mask = sex_arr[u] == sex_arr[v]
    return {
        "same": edges[:, same_mask].copy(),
        "diff": edges[:, ~same_mask].copy(),
    }


def query_attachment(
    ref_sex: "Sequence | np.ndarray",
    query_sex,
    query_index: int | None = None,
) -> dict[str, np.ndarray]:
    """One-way reference -> query edges, split by sex.

    The query occupies a single node ``query_index`` (default = len(ref_sex));
    it is never a source, so no query--query edge exists and the reference
    subgraph is left unchanged. Site never enters the signature.
    """
    ref_arr = _as_int_array(ref_sex)
    n_ref = ref_arr.shape[0]
    q = int(query_sex)
    q_idx = int(query_index) if query_index is not None else n_ref
    src = np.arange(n_ref, dtype=np.int64)
    dst = np.full(n_ref, q_idx, dtype=np.int64)
    same_mask = ref_arr == q
    return {
        "same": np.stack((src[same_mask], dst[same_mask])),
        "diff": np.stack((src[~same_mask], dst[~same_mask])),
    }
