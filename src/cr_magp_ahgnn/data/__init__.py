"""Read-only registries, splits, and fold-local preprocessing."""

from .registry import DataRegistry
from .splits import SplitManifest

__all__ = ["DataRegistry", "SplitManifest"]
