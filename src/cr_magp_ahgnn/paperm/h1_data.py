"""H1 canonical data bridge: pure-AAL 7-D node features + CC200/HO 20-D context.

Clean-room port of the authoritative recovered feature semantics
(`模型/utils/feature_eng.py`): same arctanh loading, same five weighted
topological metrics per atlas, same per-atlas mean/std context. The H1
FREEZE (docs/H1_representation_freeze.md) splits them differently from the
recovered 141-D fused feature:
  - node_features  = pure AAL, per node = its 5 AAL topological metrics
    (degree, degree_centrality, eigenvector_centrality, betweenness_centrality,
    clustering_coef) + its own arctanh AAL row mean/std  => 7-D per node.
  - topology_context = 20-D per subject: mean/std of the 5 topological metrics
    over CC200 (10) concatenated with those over HO (10).
  - AAL graph edges = |arctanh(AAL)| > sparsity_threshold (row0 = source).

Depends only on third-party libs (numpy/networkx/scipy) + frozen canonical
ABIDE-I 871 registry. Does NOT import recovered `模型/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.io as sio
import torch


def load_arctanh_mat(path) -> np.ndarray | None:
    """Load a correlation .mat and apply the recovered Fisher z transform."""
    if not Path(path).exists():
        return None
    data = sio.loadmat(str(path))
    for key in data:
        if not key.startswith("__"):
            m = np.asarray(data[key], dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                m = np.arctanh(m)
            m[np.isinf(m)] = 0.0
            m[np.isnan(m)] = 0.0
            return m
    return None


def topological_features(adj: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Replicate recovered compute_topological_features (weighted, 5 metrics)."""
    n = int(adj.shape[0])
    adj_abs = np.abs(adj)
    np.fill_diagonal(adj_abs, 0.0)
    G = nx.from_numpy_array(adj_abs)
    drop = [(u, v) for u, v, d in G.edges(data=True) if d.get("weight", 0) < threshold]
    G.remove_edges_from(drop)
    if G.number_of_nodes() == 0:
        return np.zeros((n, 5))
    deg_w = np.array([d for _, d in G.degree(weight="weight")], dtype=float)
    deg_c = np.array(list(nx.degree_centrality(G).values()), dtype=float)
    try:
        eig_c = np.array(list(nx.eigenvector_centrality(G, max_iter=500, tol=1e-4).values()), dtype=float)
    except Exception:
        eig_c = np.zeros(n)
    # EXACT betweenness (k=None): deterministic. Recovered legacy used k=20
    # approximate sampling driven by python `random` (never seeded) => NOT
    # reproducible. Exactness is required for the H1 freeze SHA.
    bet_c = np.array(list(nx.betweenness_centrality(G).values()), dtype=float)
    clus = np.array(list(nx.clustering(G, weight="weight").values()), dtype=float)
    feats = np.stack([deg_w, deg_c, eig_c, bet_c, clus], axis=1)
    return np.nan_to_num(feats)


def subject_tensors(subject_root, subject_id, threshold: float = 0.5):
    """Return (node_features 7-D, topology_context 20-D, edge_index) for one id.
    Deterministic given a fixed global np.random seed set by the caller
    (betweenness k=20 sampling is RNG-driven; recovered legacy never seeded it).
    """
    subject_root = Path(subject_root)
    aal = load_arctanh_mat(subject_root / subject_id / f"{subject_id}_aal_correlation.mat")
    cc = load_arctanh_mat(subject_root / subject_id / f"{subject_id}_cc200_correlation.mat")
    ho = load_arctanh_mat(subject_root / subject_id / f"{subject_id}_ho_correlation.mat")
    if aal is None:
        raise FileNotFoundError(f"AAL missing for {subject_id}")
    topo_aal = topological_features(aal, threshold)                      # (116,5)
    row_mean = aal.mean(axis=1, keepdims=True)                           # (116,1)
    row_std = aal.std(axis=1, keepdims=True)                             # (116,1)
    node_features = np.concatenate([topo_aal, row_mean, row_std], axis=1)  # (116,7)

    def atlas_stats(m):
        if m is None:
            return np.zeros(10)
        t = topological_features(m, threshold)                           # (n,5)
        return np.concatenate([t.mean(axis=0), t.std(axis=0)])

    context = np.concatenate([atlas_stats(cc), atlas_stats(ho)])          # (20,)
    adj_abs = np.abs(aal)
    rows, cols = np.where(adj_abs > threshold)
    keep = rows != cols
    edge_index = np.vstack((rows[keep], cols[keep])).astype(np.int64)     # (2,E)
    return (
        torch.from_numpy(node_features.astype(np.float32)),
        torch.from_numpy(context.astype(np.float32)),
        torch.from_numpy(edge_index),
    )


@dataclass
class RepresentationConfig:
    representation_id: str
    node_features_dim: int = 7
    context_features_dim: int = 20
    hidden: int = 32
    embedding_dim: int = 64
    threshold: float = 0.5
    seed: int = 666
    fold: int = 0
    split_sha256: str = ""
    subject_ids_sha256: str = ""
    canonical_sha256: dict = field(default_factory=dict)
