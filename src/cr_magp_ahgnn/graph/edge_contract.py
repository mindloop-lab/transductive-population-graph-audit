"""Direction contract shared by all clean M0-S graph operations."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EdgeContract:
    """PyG convention: row zero is source and row one is target."""

    source_row: int = 0
    target_row: int = 1
    direction: str = "source_to_target"

    def validate(self, edge_index: torch.Tensor, *, node_count: int) -> None:
        if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must be a [2, E] torch.long tensor")
        if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= node_count):
            raise ValueError("edge_index contains an out-of-range node")


PYG_EDGE_CONTRACT = EdgeContract()
