"""Custom exceptions for the public Hybrid GraphRAG API."""

from __future__ import annotations


class HybridGraphRAGError(Exception):
    """Base exception for all Hybrid GraphRAG errors."""
    pass


class ConfigurationError(HybridGraphRAGError):
    """Raised when configuration is invalid, missing, or cannot be parsed."""
    pass


class KnowledgeBaseNotFoundError(HybridGraphRAGError):
    """Raised when a specified knowledge base directory or storage cannot be found."""
    pass


class IngestionError(HybridGraphRAGError):
    """Raised when document loading, chunking, extraction, or indexing fails."""
    pass


class QueryError(HybridGraphRAGError):
    """Raised when query planning, retrieval, or answer generation fails."""
    pass


class ModelError(HybridGraphRAGError):
    """Raised when an LLM or embedding provider encounters an error or is unreachable."""
    pass
