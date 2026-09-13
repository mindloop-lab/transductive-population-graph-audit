"""H1 (B/src) subject representation loader: SubjectEncoder + canonical context.

Frozen for M3-preflight. Uses ONLY ``cr_magp_ahgnn.models.SubjectEncoder`` and
the canonical phase0d raw fixture that the Paper A campaign loads. No import of
``模型/`` or ``models/_core.py``; no recovered checkpoint initialization.

Input contract (canonical raw per-subject AAL-116 ROI graph, ``x=(116,141)``):
  aal_columns   = [0, 121]   -> per-ROI node features for the AAL GCN
  cc200_columns = [121, 131] -> summarized per subject -> context
  ho_columns    = [131, 141] -> summarized per subject -> context
Fold-local StandardScaler fitted on optimization-training rows only.
SITE_ID never enters features, edges, or gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from cr_magp_ahgnn.models import SubjectEncoder

AAL_COLS = (0, 121)
CC200_COLS = (121, 131)
HO_COLS = (131, 141)
NODE_FEATURES = AAL_COLS[1] - AAL_COLS[0]      # 121
CONTEXT_FEATURES = (CC200_COLS[1] - CC200_COLS[0]) + (HO_COLS[1] - HO_COLS[0])  # 20
HIDDEN = 32
EMBEDDING_DIM = HIDDEN * 2  # aal_embedding(32) || context(32) = 64


@dataclass
class H1SubjectContract:
    """Fold-local scalers and a small batch builder for SubjectEncoder."""

    node_scaler: StandardScaler
    context_scaler: StandardScaler
    node_dim: int = NODE_FEATURES
    context_dim: int = CONTEXT_FEATURES
    hidden: int = HIDDEN
    embedding_dim: int = EMBEDDING_DIM

    @classmethod
    def fit(cls, graphs: list[dict], train_ids: np.ndarray) -> "H1SubjectContract":
        nodes = np.concatenate([graphs[sid]["x"][:, AAL_COLS[0]:AAL_COLS[1]].numpy() for sid in train_ids], axis=0)
        ctx = np.stack([_context(graphs[sid]) for sid in train_ids])
        return cls(
            node_scaler=StandardScaler().fit(nodes),
            context_scaler=StandardScaler().fit(ctx),
        )

    def node_features(self, graphs: list[dict], sid: int) -> np.ndarray:
        return self.node_scaler.transform(graphs[sid]["x"][:, AAL_COLS[0]:AAL_COLS[1]].numpy())

    def context(self, graphs: list[dict], sid: int) -> np.ndarray:
        return self.context_scaler.transform(_context(graphs[sid]).reshape(1, -1))[0]

    def make_model(self) -> SubjectEncoder:
        return SubjectEncoder(node_features=self.node_dim, context_features=self.context_dim, hidden=self.hidden)

    def embed_subjects(
        self,
        model: SubjectEncoder,
        graphs: list[dict],
        ids: np.ndarray,
        device: torch.device,
    ) -> torch.Tensor:
        """Batch the given subjects through SubjectEncoder -> (B, embedding_dim)."""
        model.eval()
        n_per = graphs[ids[0]]["x"].shape[0]
        xs, eis, ctxs, batch, base = [], [], [], [], 0
        for pos, sid in enumerate(ids):
            x = torch.from_numpy(self.node_features(graphs, sid)).float()
            ei = graphs[sid]["edge_index"] + base
            xs.append(x); eis.append(ei)
            ctxs.append(torch.from_numpy(self.context(graphs, sid)).float())
            batch.append(torch.full((n_per,), pos, dtype=torch.long))
            base += n_per
        x = torch.cat(xs, dim=0).to(device)
        edge_index = torch.cat(eis, dim=1).to(device)
        b = torch.cat(batch, dim=0).to(device)
        ctx = torch.stack(ctxs, dim=0).to(device)
        with torch.no_grad():
            return model(x, edge_index, b, ctx)


def _context(graph: dict) -> np.ndarray:
    x = graph["x"].numpy()
    cc = x[:, CC200_COLS[0]:CC200_COLS[1]].mean(0)   # (10,)
    ho = x[:, HO_COLS[0]:HO_COLS[1]].mean(0)         # (10,)
    return np.concatenate((cc, ho))                   # (20,)


def load_raw_fixture(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)
