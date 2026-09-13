"""AAL 116-node subject path with topology and CC200/HO context."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool


class SubjectEncoder(nn.Module):
    def __init__(self, node_features: int, context_features: int, hidden: int = 32):
        super().__init__()
        self.conv1 = GCNConv(node_features, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.context = nn.Linear(context_features, hidden)
        self.output_dim = hidden * 2

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        topology_context: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.relu(self.conv1(node_features, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        aal_embedding = global_mean_pool(x, batch)
        context = torch.relu(self.context(topology_context))
        return torch.cat((aal_embedding, context), dim=1)
