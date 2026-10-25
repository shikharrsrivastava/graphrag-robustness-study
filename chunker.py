import hashlib
from collections import Counter

from config import CHUNK_OVERLAP, CHUNK_SIZE


def _split_long_paragraph(text: str, max_chars: int, overlap: int) -> list[str]:
    if overlap <= 0:
        pieces: list[str] = []
        current: list[str] = []
        current_len = 0
        for word in text.split():
            extra = len(word) if not current else 1 + len(word)
            if current and current_len + extra > max_chars:
                pieces.append(" ".join(current))
                current, current_len = [word], len(word)
            else:
                current.append(word)
                current_len += extra
        if current:
            pieces.append(" ".join(current))
        return pieces
    step = max(1, max_chars - overlap)
    pieces = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start += step
    return pieces


def chunk_records(records: list[dict]) -> list[dict]:
    """Paragraph-aware chunking with position-independent deterministic IDs.

    Each paragraph becomes exactly one chunk (oversized paragraphs are
    hard-split at word boundaries), so editing or inserting a paragraph
    never shifts the boundaries of any other chunk. A chunk ID hashes
    only (file name, occurrence counter of identical chunk text, chunk
    text) with blake2b, never the global chunk index.
    """
    chunks: list[dict] = []
    seen: Counter = Counter()
    for record in records:
        for paragraph in record["text"].split("\n\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            pieces = (
                [paragraph]
                if len(paragraph) <= CHUNK_SIZE
                else _split_long_paragraph(paragraph, CHUNK_SIZE, CHUNK_OVERLAP)
            )
            for piece in pieces:
                piece = piece.strip()
                occurrence = seen[(record["source"], piece)]
                seen[(record["source"], piece)] += 1
                chunk_id = hashlib.blake2b(
                    f"{record['source']}:{occurrence}:{piece}".encode("utf-8"),
                    digest_size=16,
                ).hexdigest()
                chunks.append(
                    {
                        "id": chunk_id,
                        "text": piece,
                        "meta": {"source": record["source"], "page": record["page"]},
                    }
                )
    return chunks
