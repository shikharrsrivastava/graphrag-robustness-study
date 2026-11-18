"""Embedding service interfacing with Ollama /api/embed and computing cosine similarities."""

from __future__ import annotations
import logging
from typing import Callable, List, Optional
import numpy as np
import requests
from src.core.configuration import OllamaConfig
from src.core.interfaces import EmbeddingProvider

logger = logging.getLogger(__name__)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between rows of a and rows of b using L2-normalization.

    Returns a matrix of shape (len(a), len(b)).
    """
    a_arr = np.atleast_2d(np.asarray(a, dtype=np.float64))
    b_arr = np.atleast_2d(np.asarray(b, dtype=np.float64))

    if a_arr.shape[1] != b_arr.shape[1]:
        raise ValueError(
            f"Dimension mismatch in cosine similarity: a has {a_arr.shape[1]} dims, b has {b_arr.shape[1]} dims"
        )

    norm_a = np.linalg.norm(a_arr, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b_arr, axis=1, keepdims=True)

    norm_a = np.where(norm_a <= 1e-12, 1e-12, norm_a)
    norm_b = np.where(norm_b <= 1e-12, 1e-12, norm_b)

    a_normalized = a_arr / norm_a
    b_normalized = b_arr / norm_b

    sim_matrix = np.dot(a_normalized, b_normalized.T)
    # Clip to [-1.0, 1.0] to account for floating point inaccuracies
    return np.clip(sim_matrix, -1.0, 1.0)


class OllamaEmbeddingService(EmbeddingProvider):
    """Generates dense embeddings via Ollama /api/embed with strict dimension and finiteness checks."""

    def __init__(
        self,
        config: Optional[OllamaConfig] = None,
        custom_embedder: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ) -> None:
        self.config = config or OllamaConfig()
        self.custom_embedder = custom_embedder

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        results = self.embed_batch([text])
        if not results:
            raise ValueError(f"Failed to generate embedding for text: '{text[:50]}...'")
        return results[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings, validating dimensions, finiteness, and norms."""
        if not texts:
            return []

        if self.custom_embedder is not None:
            raw_embeddings = self.custom_embedder(texts)
        else:
            url = f"{self.config.base_url.rstrip('/')}/api/embed"
            payload = {
                "model": self.config.embedding_model,
                "input": texts,
            }
            response = requests.post(
                url, json=payload, timeout=self.config.timeout_seconds
            )
            response.raise_for_status()
            data = response.json()
            raw_embeddings = data.get("embeddings", [])

        if len(raw_embeddings) != len(texts):
            raise ValueError(
                f"Expected {len(texts)} embeddings from provider, but received {len(raw_embeddings)}"
            )

        validated: List[List[float]] = []
        for idx, vec in enumerate(raw_embeddings):
            # 1. Dimension validation
            if len(vec) != self.config.expected_embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch at index {idx}: expected {self.config.expected_embedding_dim}, got {len(vec)}"
                )

            arr = np.asarray(vec, dtype=np.float64)

            # 2. Finiteness validation (no NaN or Inf)
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"Embedding at index {idx} contains non-finite values (NaN or Inf)."
                )

            # 3. Non-zero norm validation
            norm = float(np.linalg.norm(arr))
            if norm <= 1e-12:
                raise ValueError(
                    f"Embedding at index {idx} has zero or near-zero L2 norm: {norm}"
                )

            validated.append([float(x) for x in vec])

        return validated
