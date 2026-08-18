"""Optional reranker hook; baseline never imports torch or requires model weights."""
from __future__ import annotations

from pathlib import Path


def available(weights_path: str | Path = "models/reranker_weights.pt") -> bool:
    return Path(weights_path).exists()


def rerank(*args, **kwargs):
    """Reserved extension point. Raises a clear error instead of silently changing baseline behavior."""
    raise RuntimeError("Optional reranker weights are not present or the extension is not enabled; use the deterministic baseline.")
