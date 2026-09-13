"""Clean-room baseline adapters for frozen external algorithm specifications."""

from .ev_gcn_adapter import EVGCNAdapter
from .parisot_gcn_adapter import ParisotGCNAdapter
from .single_subject_popgnn_adapter import SingleSubjectPopGNNAdapter

__all__ = ["EVGCNAdapter", "ParisotGCNAdapter", "SingleSubjectPopGNNAdapter"]
