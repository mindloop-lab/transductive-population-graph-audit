"""H1: site-free heterogeneous population head + inductive strict trainer.

Paper M / MedIA gate H1. This is the TRAINING MACHINERY (M2). The population
head mirrors the recovered heterogeneous `PopulationGNNCore` semantics
(`模型/models/_core.py`): per layer a same-sex and a cross-sex TransformerConv
channel whose outputs are softmax-weighted per layer, with BatchNorm and leaky
ReLU. It consumes subject embeddings (in production, the recovered
``SubjectGNN`` output) and SITE-free SEX relation edges produced by
``h1_edges``.

Everything here is inductive: the population forward over the reference
(optimization-train) subgraph only; each out-of-fold subject is attached as a
single query with one-way reference->query edges (never query--query), so the
reference representation is untouched by evaluation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import TransformerConv


class H1PopulationHead(nn.Module):
    """Heterogeneous same/cross-sex population reasoning, mirroring recovered
    PopulationGNNCore (3 layers, softmax relation fusion, leaky ReLU, BN)."""

    def __init__(self, in_dim: int = 64, classes: int = 2, dropout: float = 0.3):  # 64 = freeze embedding
        super().__init__()
        h = in_dim
        self.dropout = dropout
        self.conv1_same = TransformerConv(in_dim, h, heads=1)
        self.conv1_diff = TransformerConv(in_dim, h, heads=1)
        self.bn1 = nn.BatchNorm1d(h)
        self.conv2_same = TransformerConv(h, h, heads=1)
        self.conv2_diff = TransformerConv(h, h, heads=1)
        self.bn2 = nn.BatchNorm1d(h)
        self.conv3_same = TransformerConv(h, h, heads=1)
        self.conv3_diff = TransformerConv(h, h, heads=1)
        self.bn3 = nn.BatchNorm1d(h)
        self.classifier = nn.Linear(h, classes)
        self.weights = nn.Parameter(torch.ones(3, 2))

    def _layer(self, x, same, diff, conv_same, conv_diff, bn, weights, training):
        w = torch.softmax(self.weights[weights], dim=0)
        h_same = conv_same(x, same)
        h_diff = conv_diff(x, diff)
        x = w[0] * h_same + w[1] * h_diff
        x = bn(x)
        return torch.nn.functional.leaky_relu(x)

    def forward(self, x: torch.Tensor, same_idx: torch.Tensor, diff_idx: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self._layer(x, same_idx, diff_idx, self.conv1_same, self.conv1_diff, self.bn1, 0, self.training)
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self._layer(x, same_idx, diff_idx, self.conv2_same, self.conv2_diff, self.bn2, 1, self.training)
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self._layer(x, same_idx, diff_idx, self.conv3_same, self.conv3_diff, self.bn3, 2, self.training)
        return self.classifier(x)


def _as_tensor(edges: dict[str, np.ndarray], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    same = torch.from_numpy(edges["same"]).long().to(device)
    diff = torch.from_numpy(edges["diff"]).long().to(device)
    return same, diff


def reference_logits(
    head: H1PopulationHead,
    ref_embeddings: torch.Tensor,
    ref_edges: dict[str, np.ndarray],
) -> torch.Tensor:
    """Population pass over the frozen reference subgraph only."""
    same, diff = _as_tensor(ref_edges, ref_embeddings.device)
    return head(ref_embeddings, same, diff)


def full_forward(
    head: H1PopulationHead,
    ref_embeddings: torch.Tensor,
    ref_edges: dict[str, np.ndarray],
    query_embedding: torch.Tensor,
    query_edges: dict[str, np.ndarray],
) -> torch.Tensor:
    """One forward over ``ref + [query]`` with edges
    ``ref_internal + reference->query`` (query is never a source)."""
    n_ref = ref_embeddings.shape[0]
    x = torch.cat((ref_embeddings, query_embedding.reshape(1, -1)), dim=0)
    ref_same, ref_diff = _as_tensor(ref_edges, x.device)
    q_same, q_diff = _as_tensor(query_edges, x.device)
    same = torch.cat((ref_same, q_same), dim=1)
    diff = torch.cat((ref_diff, q_diff), dim=1)
    return head(x, same, diff)


def attach_query(
    head: H1PopulationHead,
    ref_embeddings: torch.Tensor,
    ref_edges: dict[str, np.ndarray],
    query_embedding: torch.Tensor,
    query_edges: dict[str, np.ndarray],
) -> torch.Tensor:
    """One-way single-query inference: returns only the query logit.

    Because the query is never a source, the reference rows of ``full_forward``
    are bit-identical to a reference-only pass (see ``full_forward``)."""
    n_ref = ref_embeddings.shape[0]
    logits = full_forward(head, ref_embeddings, ref_edges, query_embedding, query_edges)
    return logits[n_ref : n_ref + 1]


class LabelSmoothCE(nn.Module):
    def __init__(self, smoothing: float = 0.1, classes: int = 2):
        super().__init__()
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.reshape(-1)
        one_hot = torch.zeros_like(logits).scatter(1, target.unsqueeze(1), 1.0)
        smooth = one_hot * (1 - self.smoothing) + (1 - one_hot) * self.smoothing / (self.classes - 1)
        log_p = torch.log_softmax(logits, dim=1)
        return torch.nn.functional.kl_div(log_p, smooth, reduction="none").sum(1).mean()


def train_inductive(
    head: H1PopulationHead,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    ref_embeddings: torch.Tensor,
    ref_sex: np.ndarray,
    ref_labels: torch.Tensor,
    epochs: int,
    seed: int = 0,
) -> list[float]:
    """Train the population head inductively on the frozen reference subgraph."""
    from . import h1_edges

    torch.manual_seed(seed)
    edges = h1_edges.reference_relations(ref_sex)
    same, diff = _as_tensor(edges, ref_embeddings.device)
    losses: list[float] = []
    for _ in range(epochs):
        head.train()
        optimizer.zero_grad()
        logits = head(ref_embeddings, same, diff)
        loss = loss_fn(logits, ref_labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def oof_single_query(
    head: H1PopulationHead,
    ref_embeddings: torch.Tensor,
    ref_sex: np.ndarray,
    query_embeddings: torch.Tensor,
    query_sex: np.ndarray,
    query_ids: np.ndarray,
) -> dict[str, np.ndarray | torch.Tensor]:
    """Strict single-query OOF: each query attached one at a time; returns the
    per-query logits in the supplied query_id order."""
    from . import h1_edges

    head.eval()
    device = ref_embeddings.device
    ref_edges = h1_edges.reference_relations(ref_sex)
    logits = []
    with torch.no_grad():
        for k in range(query_embeddings.shape[0]):
            qe = query_embeddings[k]
            qs = int(query_sex[k])
            q_edges = h1_edges.query_attachment(ref_sex, qs, query_index=ref_embeddings.shape[0])
            out = attach_query(head, ref_embeddings, ref_edges, qe, q_edges)
            logits.append(out)
    logits_t = torch.cat(logits, dim=0).detach().cpu()
    return {"subject_ids": np.asarray(query_ids), "logits": logits_t}
