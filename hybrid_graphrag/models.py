"""Public data models for Hybrid GraphRAG queries and results."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field
from src.core.models import RetrievalResult


class Citation(BaseModel):
    """Represents a verified evidence citation for an answer."""
    model_config = ConfigDict(extra="ignore")

    chunk_id: str = Field(..., min_length=1, description="Unique chunk identifier")
    doc_id: str = Field(default="", description="Source document identifier")
    filename: str = Field(default="", description="Original document filename")
    page_numbers: List[int] = Field(default_factory=list, description="Source page numbers")
    text_snippet: Optional[str] = Field(default=None, description="Excerpt of the cited chunk text")
    valid: bool = Field(default=True, description="Whether the citation is verified against retrieved chunks")

    @property
    def citation_id(self) -> str:
        """Alias for chunk_id for backward compatibility."""
        return self.chunk_id

    @property
    def text(self) -> Optional[str]:
        """Alias for text_snippet for backward compatibility."""
        return self.text_snippet

    def __str__(self) -> str:
        parts = [self.chunk_id]
        if self.filename or self.doc_id:
            parts.append(f"Doc: {self.filename or self.doc_id}")
        if self.page_numbers:
            parts.append(f"p.{', '.join(str(p) for p in self.page_numbers)}")
        return f"[{' | '.join(parts)}]"


class QueryResult(BaseModel):
    """Structured result returned by HybridGraphRAG.query()."""
    model_config = ConfigDict(extra="ignore")

    question: str
    answer: str
    canonical_answer: Optional[str] = None
    abstained: bool = False
    abstention_reason: Optional[str] = None
    citations: List[Citation] = Field(default_factory=list)
    cited_chunk_ids: List[str] = Field(default_factory=list)
    mode: str = "hybrid"
    latency_seconds: float = 0.0
    timing: Dict[str, float] = Field(default_factory=dict)
    raw_result: Optional[RetrievalResult] = None

    @property
    def route(self) -> str:
        """Alias for mode for backward compatibility."""
        return self.mode

    def __str__(self) -> str:
        if self.abstained:
            return f"[Abstained] {self.answer}"
        if self.canonical_answer:
            return f"[{self.canonical_answer}] {self.answer}"
        return self.answer
