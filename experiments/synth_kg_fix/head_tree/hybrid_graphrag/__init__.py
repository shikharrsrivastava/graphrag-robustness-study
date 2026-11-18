"""Hybrid GraphRAG: Domain-Agnostic Enterprise Knowledge Graph & Vector Retrieval Engine."""

from __future__ import annotations

from hybrid_graphrag.api import HybridGraphRAG
from hybrid_graphrag.models import Citation, QueryResult
from hybrid_graphrag.exceptions import (
    ConfigurationError,
    HybridGraphRAGError,
    IngestionError,
    KnowledgeBaseNotFoundError,
    ModelError,
    QueryError,
)

__version__ = "1.0.0"

__all__ = [
    "HybridGraphRAG",
    "QueryResult",
    "Citation",
    "HybridGraphRAGError",
    "ConfigurationError",
    "KnowledgeBaseNotFoundError",
    "IngestionError",
    "QueryError",
    "ModelError",
]
