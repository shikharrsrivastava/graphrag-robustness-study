"""Protocol interfaces separating the reusable GraphRAG core from concrete implementations."""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Protocol, Tuple
from src.core.models import (
    Chunk,
    Document,
    EntityMention,
    Evidence,
    Page,
    QueryPlan,
    Relationship,
    RetrievalResult,
)


class DocumentLoader(Protocol):
    def load_directory(self, input_dir: Path, kb_id: str) -> Tuple[List[Document], List[Page]]:
        ...


class TextChunker(Protocol):
    def chunk_pages(self, pages: List[Page], kb_id: str) -> List[Chunk]:
        ...


class EntityExtractor(Protocol):
    def extract_entities(self, chunk: Chunk) -> List[EntityMention]:
        ...


class RelationshipExtractor(Protocol):
    def extract_relationships(
        self, chunk: Chunk, mentions: List[EntityMention]
    ) -> List[Relationship]:
        ...


class EmbeddingProvider(Protocol):
    def embed_text(self, text: str) -> List[float]:
        ...

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        ...


class Retriever(Protocol):
    def retrieve(
        self, question: str, plan: QueryPlan, kb_id: str, scope_id: Optional[str] = None
    ) -> List[Evidence]:
        ...


class AnswerGenerator(Protocol):
    def generate_answer(self, retrieval_result: RetrievalResult) -> str:
        ...
