"""Deterministic text chunking implementation preserving sentence boundaries and provenance."""

from __future__ import annotations
from collections import defaultdict
import hashlib
import logging
from typing import Dict, List, Optional
from src.core.configuration import ChunkingConfig
from src.core.interfaces import TextChunker
from src.core.models import Chunk, Page

logger = logging.getLogger(__name__)


class DeterministicChunker(TextChunker):
    """Chunks document pages deterministically according to ChunkingConfig.

    Tracks multi-page provenance spans and respects boundary hierarchy
    (paragraph -> newline -> sentence punctuation -> hard cut).
    """

    def __init__(self, config: Optional[ChunkingConfig] = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk_pages(self, pages: List[Page], kb_id: str) -> List[Chunk]:
        """Group pages by doc_id and generate deterministic Chunk objects."""
        # Group pages by doc_id, preserving encounter order of documents
        grouped_pages: Dict[str, List[Page]] = defaultdict(list)
        for page in pages:
            # Enforce KB matching
            if page.kb_id != kb_id:
                logger.warning(
                    "Page kb_id '%s' does not match target kb_id '%s', skipping.",
                    page.kb_id,
                    kb_id,
                )
                continue
            grouped_pages[page.doc_id].append(page)

        all_chunks: List[Chunk] = []

        # Sort doc_ids deterministically
        for doc_id in sorted(grouped_pages.keys()):
            doc_pages = grouped_pages[doc_id]
            # Sort pages ascending by page_number
            doc_pages.sort(key=lambda p: p.page_number)

            # Skip empty pages
            non_empty_pages = [p for p in doc_pages if not p.is_empty and len(p.text.strip()) > 0]
            if not non_empty_pages:
                continue

            doc_chunks = self._chunk_document_pages(
                doc_id=doc_id,
                pages=non_empty_pages,
                kb_id=kb_id,
            )
            all_chunks.extend(doc_chunks)

        return all_chunks

    def _chunk_document_pages(
        self, doc_id: str, pages: List[Page], kb_id: str
    ) -> List[Chunk]:
        """Concatenate document pages and slide the chunking window."""
        full_text = ""
        # List of (start_char, end_char, page_number)
        page_spans: List[tuple[int, int, int]] = []
        filename = pages[0].filename

        for i, page in enumerate(pages):
            if i > 0:
                full_text += "\n\n"
            start_char = len(full_text)
            full_text += page.text
            end_char = len(full_text)
            page_spans.append((start_char, end_char, page.page_number))

        total_len = len(full_text)
        if total_len == 0:
            return []

        chunks: List[Chunk] = []
        start_idx = 0
        chunk_index = 0

        while start_idx < total_len:
            # If remaining text fits within chunk_size, take the rest
            if total_len - start_idx <= self.config.chunk_size:
                actual_end = total_len
            else:
                max_end = start_idx + self.config.chunk_size
                actual_end = self._find_boundary_cut(full_text, start_idx, max_end)

            raw_slice = full_text[start_idx:actual_end]
            chunk_text = raw_slice.strip()

            if chunk_text:
                # Determine which pages this chunk spans
                spanned_pages: List[int] = []
                for p_start, p_end, p_num in page_spans:
                    if max(start_idx, p_start) < min(actual_end, p_end):
                        spanned_pages.append(p_num)

                if not spanned_pages:
                    # Defensive fallback to the first page overlapping start_idx
                    spanned_pages = [page_spans[0][2]]

                spanned_pages = sorted(list(set(spanned_pages)))
                chunk_id = f"{doc_id}_C{chunk_index:04d}"
                chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()

                chunk = Chunk(
                    kb_id=kb_id,
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    filename=filename,
                    page_numbers=spanned_pages,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    char_count=len(chunk_text),
                    content_hash=chunk_hash,
                    metadata={
                        "start_char": start_idx,
                        "end_char": actual_end,
                    },
                )
                chunks.append(chunk)
                chunk_index += 1

            if actual_end >= total_len:
                break

            # Advance window using chunk_overlap and guarantee forward progress
            next_start = actual_end - self.config.chunk_overlap
            if next_start <= start_idx:
                next_start = start_idx + 1
            start_idx = next_start

        return chunks

    def _find_boundary_cut(self, full_text: str, start_idx: int, max_end: int) -> int:
        """Find boundary cut in the last 25% of the window respecting hierarchy."""
        if not self.config.respect_sentence_boundaries:
            return max_end

        search_len = int(self.config.chunk_size * 0.25)
        search_start = max(start_idx, max_end - search_len)
        search_text = full_text[search_start:max_end]

        # 1. Paragraph break (\n\n)
        p_pos = search_text.rfind("\n\n")
        if p_pos != -1:
            return search_start + p_pos + 2

        # 2. Single newline (\n)
        n_pos = search_text.rfind("\n")
        if n_pos != -1:
            return search_start + n_pos + 1

        # 3. Sentence punctuation ('. ', '? ', '! ')
        s_pos = -1
        for punct in (". ", "? ", "! "):
            pos = search_text.rfind(punct)
            if pos > s_pos:
                s_pos = pos
        if s_pos != -1:
            return search_start + s_pos + 1

        # Fallback to hard cut
        return max_end
