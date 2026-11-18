"""Document-metadata-based seed discovery mechanism.

Identifies candidate documents whose authoritative titles match question-specified
entity references using conservative, exact and punctuation-normalized matching.
"""

from __future__ import annotations
import logging
import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, ConfigDict, Field
from src.core.configuration import RetrievalConfig
from src.core.models import SeedEntitySpec
from src.storage.document_store import DocumentStore

logger = logging.getLogger(__name__)

GENERIC_TITLE_STOPWORDS: Set[str] = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "by",
    "from",
    "about",
    "into",
    "through",
    "after",
    "over",
    "between",
    "out",
    "against",
    "during",
    "without",
    "before",
    "under",
    "around",
    "among",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "document",
    "doc",
    "article",
    "page",
    "entry",
    "chapter",
    "section",
    "part",
}


def normalize_title(text: str) -> str:
    """Normalize title with Unicode NFKC, case folding, and collapsed whitespace."""
    if not text:
        return ""
    norm = unicodedata.normalize("NFKC", text).strip().casefold()
    return re.sub(r"\s+", " ", norm)


def punctuation_normalized_title(text: str) -> str:
    """Normalize title by replacing punctuation with spaces and collapsing whitespace."""
    norm = normalize_title(text)
    norm = re.sub(r"[^\w\s]", " ", norm)
    return re.sub(r"\s+", " ", norm).strip()


def strip_leading_article(text: str) -> str:
    """Strip leading grammatical article ('the', 'a', 'an') from normalized title."""
    tokens = text.split()
    if tokens and tokens[0] in ("the", "a", "an"):
        return " ".join(tokens[1:])
    return text


def is_generic_title(text: str) -> bool:
    """Check if a title is overly broad, generic, or consists entirely of stopwords."""
    norm = punctuation_normalized_title(text)
    if not norm or len(norm) < 2:
        return True
    tokens = norm.split()
    if not tokens:
        return True
    if len(tokens) == 1 and (tokens[0] in GENERIC_TITLE_STOPWORDS or len(tokens[0]) <= 1):
        return True
    if all(t in GENERIC_TITLE_STOPWORDS for t in tokens):
        return True
    return False


class DocumentTitleCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed_name: str = Field(..., min_length=1, description="Original seed entity query string")
    matched_title: str = Field(..., min_length=1, description="Authoritative matched document title")
    doc_id: str = Field(..., min_length=1, description="Matched document identifier")
    chunk_ids: List[str] = Field(default_factory=list, description="Discovered chunk IDs for this document")
    matching_method: str = Field(
        ...,
        description="EXACT_NORMALIZED, PUNCTUATION_NORMALIZED, or ARTICLE_NORMALIZED",
    )
    is_ambiguous: bool = Field(default=False, description="True if multiple documents shared the normalized title")
    candidate_doc_ids: List[str] = Field(
        default_factory=list,
        description="All document IDs matching the title if ambiguous",
    )


class DocumentTitleDiscoverer:
    """Conservative, document-metadata-based entity discovery engine.
    
    Finds documents whose authoritative titles match seed specifications,
    enforcing exact/punctuation matching, generic title protection,
    ambiguity tracking, and strict candidate limits.
    """

    def __init__(
        self,
        doc_store: DocumentStore,
        config: Optional[RetrievalConfig] = None,
        max_title_matches_per_seed: int = 2,
        max_chunks_per_title_match: int = 3,
        max_title_candidates_per_query: int = 6,
    ) -> None:
        self.doc_store = doc_store
        if config is not None:
            self.max_title_matches_per_seed = config.max_title_matches_per_seed
            self.max_chunks_per_title_match = config.max_chunks_per_title_match
            self.max_title_candidates_per_query = config.max_title_candidates_per_query
        else:
            self.max_title_matches_per_seed = max_title_matches_per_seed
            self.max_chunks_per_title_match = max_chunks_per_title_match
            self.max_title_candidates_per_query = max_title_candidates_per_query

    def discover_title_candidates(
        self,
        seed_specs: List[SeedEntitySpec],
        scope_id: Optional[str] = None,
    ) -> List[DocumentTitleCandidate]:
        """Discover candidate documents and chunks matching query seed entity specifications.
        
        Args:
            seed_specs: List of SeedEntitySpec items from QueryPlan.
            scope_id: Optional scope filter for multi-tenant isolation.
            
        Returns:
            List of DocumentTitleCandidate objects within configured bounds.
        """
        if not seed_specs:
            return []

        doc_titles = self.doc_store.load_document_titles()
        if not doc_titles:
            return []

        # Precompute normalized versions of authoritative document titles
        # Map: doc_id -> (authoritative_title, exact_norm, punct_norm, article_norm)
        title_cache: Dict[str, Tuple[str, str, str, str]] = {}
        for d_id, raw_title in doc_titles.items():
            exact_norm = normalize_title(raw_title)
            punct_norm = punctuation_normalized_title(raw_title)
            article_norm = strip_leading_article(punct_norm)
            title_cache[d_id] = (raw_title, exact_norm, punct_norm, article_norm)

        results: List[DocumentTitleCandidate] = []
        total_chunks_admitted = 0

        for spec in seed_specs:
            seed_name = spec.name.strip()
            if not seed_name or is_generic_title(seed_name):
                logger.debug("Skipping generic or empty seed spec: '%s'", seed_name)
                continue

            s_exact = normalize_title(seed_name)
            s_punct = punctuation_normalized_title(seed_name)
            s_article = strip_leading_article(s_punct)

            matched_docs: List[Tuple[str, str, str]] = []  # (doc_id, matched_title, method)

            for d_id, (raw_title, d_exact, d_punct, d_article) in title_cache.items():
                if is_generic_title(raw_title):
                    continue

                method: Optional[str] = None
                if s_exact == d_exact:
                    method = "EXACT_NORMALIZED"
                elif s_punct == d_punct:
                    method = "PUNCTUATION_NORMALIZED"
                elif s_article and d_article and s_article == d_article:
                    method = "ARTICLE_NORMALIZED"

                if method is not None:
                    matched_docs.append((d_id, raw_title, method))

            if not matched_docs:
                continue

            # Sort deterministically by doc_id
            matched_docs.sort(key=lambda x: x[0])

            is_ambig = len(matched_docs) > 1
            all_candidate_ids = [m[0] for m in matched_docs]

            admitted_docs = matched_docs[: self.max_title_matches_per_seed]

            for d_id, matched_title, method in admitted_docs:
                chunks = self.doc_store.get_chunks_for_doc(d_id, scope_id=scope_id)
                # Sort deterministically by chunk_index
                chunks.sort(key=lambda c: c.chunk_index)

                available_chunk_ids = [c.chunk_id for c in chunks]
                capped_chunk_ids = available_chunk_ids[: self.max_chunks_per_title_match]

                # Check query-level cap
                remaining_quota = max(0, self.max_title_candidates_per_query - total_chunks_admitted)
                if remaining_quota <= 0:
                    logger.debug("Document title query candidate quota reached (%d)", self.max_title_candidates_per_query)
                    break

                admitted_chunk_ids = capped_chunk_ids[:remaining_quota]
                total_chunks_admitted += len(admitted_chunk_ids)

                results.append(
                    DocumentTitleCandidate(
                        seed_name=seed_name,
                        matched_title=matched_title,
                        doc_id=d_id,
                        chunk_ids=admitted_chunk_ids,
                        matching_method=method,
                        is_ambiguous=is_ambig,
                        candidate_doc_ids=all_candidate_ids,
                    )
                )

                if total_chunks_admitted >= self.max_title_candidates_per_query:
                    break

            if total_chunks_admitted >= self.max_title_candidates_per_query:
                break

        return results
