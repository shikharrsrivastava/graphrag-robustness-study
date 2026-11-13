"""Hybrid GraphRAG Core Package."""

from typing import Any

__all__ = ["HybridGraphRAG", "QueryResult", "Citation"]


def __getattr__(name: str) -> Any:
    if name in ("HybridGraphRAG", "QueryResult", "Citation"):
        import hybrid_graphrag
        return getattr(hybrid_graphrag, name)
    raise AttributeError(f"module 'src' has no attribute {name!r}")
