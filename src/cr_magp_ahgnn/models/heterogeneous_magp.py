"""Clean-room heterogeneous MAGP-AHGNN (transductive model-paper lineage).

Faithful re-implementation of the RECOVERED heterogeneous MAGP-AHGNN core
(模型/models/_core.py: AAL-116 SubjectGNN with SAGPool+DiffPool, plus a
sex-aware dual-channel TransformerConv population GNN) so that:

  * it is importable WITHOUT the recovered legacy package, argparse side
    effects, or ConnectomeStreamer;
  * every submodule and parameter name matches the recovered checkpoint layout
    (subject_encoder.*, population_gnn.*), so a recovered checkpoint can be
    loaded with `load_state_dict(strict=True)` as an equivalence oracle;
  * nothing here imports legacy code or touches canonical ABIDE data.

The recovered core is the immutable semantic oracle; this module is the clean
reference used for transductive retraining and for the model-paper experiments.
Numerics are intended to match the recovered forward exactly (same
torch_geometric ops, same layer order, same pooling/DiffPool/BN semantics).

NOT the same lineage as `models/m0_strict.py` (homogeneous mean-aggregate
strict-single-query analog, which must not be presented as MAGP-AHGNN).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGPooling, ChebConv, TransformerConv
from torch_geometric.nn import dense_diff_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj


@dataclass(frozen=True)
class MAGPConfig:
    num_nodes: int = 116
    in_dim: int = 32            # post fold-PCA subject node features
    hidden_dim: int = 64
    embed_dim: int = 32
    pool_ratio_sparse: float = 0.6925   # SAGPooling ratio
    pool_ratio_dense: float = 0.3894    # DiffPool cluster ratio
    dropout: float = 0.3
    num_classes: int = 2
    device: str = "cuda:0"


class SubjectEncoder(nn.Module):
    """Individual AAL-116 graph encoder: 3x GCN + SAGPool + DiffPool fusion -> embed_dim."""

    def __init__(self, cfg: MAGPConfig):
        super().__init__()
        self.cfg = cfg
        self.num_nodes = cfg.num_nodes
        self.in_dim = cfg.in_dim
        self.hidden_dim = cfg.hidden_dim
        self.embed_dim = cfg.embed_dim
        self.gcn1 = GCNConv(cfg.in_dim, cfg.hidden_dim)
        self.gcn2 = GCNConv(cfg.hidden_dim, cfg.hidden_dim)
        self.gcn3 = GCNConv(cfg.hidden_dim, cfg.embed_dim)
        self.sag_pool = SAGPooling(cfg.embed_dim, ratio=cfg.pool_ratio_sparse)
        self.num_clusters = int(cfg.num_nodes * cfg.pool_ratio_dense)
        self.diff_pool_score = ChebConv(cfg.embed_dim, self.num_clusters, K=3)

    def forward(self, x, edge_index, edge_attr, batch, adj=None, device=None):
        if device is not None and x.device.type != "cuda" and str(device).startswith("cuda"):
            x = x.to(device); edge_index = edge_index.to(device)
            if edge_attr is not None: edge_attr = edge_attr.to(device)
            if batch is not None: batch = batch.to(device)
            if adj is not None: adj = adj.to(device)

        x = self.gcn1(x, edge_index, edge_attr)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.cfg.dropout, training=self.training)

        x = self.gcn2(x, edge_index, edge_attr)
        x = F.leaky_relu(x)

        x = self.gcn3(x, edge_index, edge_attr)
        x = F.leaky_relu(x)

        x_sag, _, _, batch_sag, _, _ = self.sag_pool(x, edge_index, edge_attr=edge_attr, batch=batch)
        embed_sag = global_mean_pool(x_sag, batch_sag)

        x_dense, _ = to_dense_batch(x, batch)
        if adj is not None:
            adj_dense = adj
            if adj_dense.dim() == 2:
                adj_dense = adj_dense.unsqueeze(0)
        else:
            adj_dense = to_dense_adj(edge_index, batch, max_num_nodes=self.num_nodes)

        s = self.diff_pool_score(x, edge_index)
        s_dense, _ = to_dense_batch(s, batch)
        s_dense = F.softmax(s_dense, dim=-1)

        # DiffPool link/entropy losses intentionally unused (matches recovered core)
        x_diff, _, _, _ = dense_diff_pool(x_dense, adj_dense, s_dense)
        embed_diff = torch.mean(x_diff, dim=1)

        return embed_sag + embed_diff


class PopulationHead(nn.Module):
    """Sex-aware heterogeneous population GNN: dual-channel TransformerConv."""

    def __init__(self, cfg: MAGPConfig, in_dim: int = 32):
        super().__init__()
        self.cfg = cfg
        self.conv1_same = TransformerConv(in_dim, 32, heads=1)
        self.conv1_diff = TransformerConv(in_dim, 32, heads=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2_same = TransformerConv(32, 32, heads=1)
        self.conv2_diff = TransformerConv(32, 32, heads=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3_same = TransformerConv(32, 32, heads=1)
        self.conv3_diff = TransformerConv(32, 32, heads=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.lin = nn.Linear(32, cfg.num_classes)
        self.weights = nn.Parameter(torch.ones(3, 2))

    def forward(self, x, same_idx, diff_idx):
        w = F.softmax(self.weights, dim=1)
        x = F.dropout(x, p=self.cfg.dropout, training=self.training)
        h_s = self.conv1_same(x, same_idx); h_d = self.conv1_diff(x, diff_idx)
        x = w[0, 0] * h_s + w[0, 1] * h_d
        x = self.bn1(x); x = F.leaky_relu(x)

        x = F.dropout(x, p=self.cfg.dropout, training=self.training)
        h_s = self.conv2_same(x, same_idx); h_d = self.conv2_diff(x, diff_idx)
        x = w[1, 0] * h_s + w[1, 1] * h_d
        x = self.bn2(x); x = F.leaky_relu(x)

        x = F.dropout(x, p=self.cfg.dropout, training=self.training)
        h_s = self.conv3_same(x, same_idx); h_d = self.conv3_diff(x, diff_idx)
        x = w[2, 0] * h_s + w[2, 1] * h_d
        x = self.bn3(x); x = F.leaky_relu(x)

        return self.lin(x)


class HeterogeneousMAGP(nn.Module):
    """Clean full model: subject_encoder -> population_gnn (transductive).

    State-dict keys subject_encoder.* and population_gnn.* match the recovered
    MAGP-AHGNN checkpoint layout, so recovered checkpoints load strict=True.
    """

    def __init__(self, cfg: MAGPConfig):
        super().__init__()
        self.cfg = cfg
        self.subject_encoder = SubjectEncoder(cfg)
        self.population_gnn = PopulationHead(cfg, in_dim=cfg.embed_dim)
        self._device = torch.device(cfg.device)

    def encode_subjects(self, graphs) -> torch.Tensor:
        """graphs: iterable of objects with .x,.edge_index,.edge_attr,.batch,(.adj)."""
        embeddings = []
        for g in graphs:
            batch = getattr(g, "batch", None)
            if batch is None:
                batch = torch.zeros(g.num_nodes, dtype=torch.long, device=g.x.device)
            adj = getattr(g, "adj", None)
            embeddings.append(self.subject_encoder(
                g.x, g.edge_index, g.edge_attr, batch, adj, device=str(self._device)))
        return torch.cat(embeddings, dim=0)

    def encode_subjects_batched(self, graphs) -> torch.Tensor:
        """Same result as encode_subjects but ONE forward over a single graph that
        concatenates all subjects (each 116-node graph separated by batch labels),
        preserving each subject's dense adjacency for the DiffPool branch so the
        numerics match the per-subject loop. Returns (N, embed_dim)."""
        xs, es, eas, adj_s = [], [], [], []
        offset = 0
        for gi, g in enumerate(graphs):
            n = g.num_nodes
            xs.append(g.x)
            ei = g.edge_index.clone()
            ei = ei + offset
            es.append(ei)
            if g.edge_attr is not None:
                eas.append(g.edge_attr)
            adj_s.append(g.adj)
            offset += n
        x_all = torch.cat(xs, dim=0)
        edge_all = torch.cat(es, dim=1)
        edge_attr_all = torch.cat(eas, dim=0) if eas else None
        batch_all = torch.cat([torch.full((g.num_nodes,), i, dtype=torch.long,
                                          device=g.x.device) for i, g in enumerate(graphs)], dim=0)
        adj_stack = torch.stack([a.to(x_all.device) if a.dim() == 2 else a for a in adj_s], dim=0)
        return self.subject_encoder(x_all, edge_all, edge_attr_all, batch_all, adj_stack,
                                    device=str(self._device))

    def forward(self, graphs, same_idx, diff_idx):
        all_embeddings = self.encode_subjects(graphs)
        return self.population_gnn(all_embeddings, same_idx, diff_idx)
