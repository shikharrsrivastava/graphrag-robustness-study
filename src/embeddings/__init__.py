"""Embeddings package providing embedding services and similarity functions."""

from src.embeddings.embedding_service import (
    OllamaEmbeddingService,
    cosine_similarity_matrix,
)

__all__ = ["OllamaEmbeddingService", "cosine_similarity_matrix"]
