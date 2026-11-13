"""Comprehensive unit tests for PDFLoader, DeterministicChunker, and DocumentStore."""

from __future__ import annotations
import hashlib
from pathlib import Path
from typing import List, Optional
import pypdf
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
import pytest
from src.core.configuration import ChunkingConfig
from src.core.models import Chunk, Document, Page
from src.ingestion.chunker import DeterministicChunker
from src.ingestion.pdf_loader import PDFLoader
from src.storage.document_store import DocumentStore


def create_test_pdf(file_path: Path, pages: List[Optional[str]]) -> Path:
    """Helper to generate a multi-page PDF with optional text content without reportlab."""
    writer = pypdf.PdfWriter()
    for text in pages:
        if text is None or not text.strip():
            writer.add_blank_page(width=612, height=792)
        else:
            page = writer.add_blank_page(width=612, height=792)
            font = DictionaryObject()
            font[NameObject("/Type")] = NameObject("/Font")
            font[NameObject("/Subtype")] = NameObject("/Type1")
            font[NameObject("/BaseFont")] = NameObject("/Helvetica")
            font_res = DictionaryObject()
            font_res[NameObject("/F1")] = font
            resources = DictionaryObject()
            resources[NameObject("/Font")] = font_res
            page[NameObject("/Resources")] = resources

            lines = text.split("\n")
            stream_parts = ["BT", "/F1 12 Tf", "14 TL", "72 720 Td"]
            for idx, line in enumerate(lines):
                safe = line.replace("\\", "\\\\").replace("(", r"\(").replace(")", r"\)")
                if idx == 0:
                    stream_parts.append(f"({safe}) Tj")
                else:
                    stream_parts.append(f"T* ({safe}) Tj")
            stream_parts.append("ET")
            stream_data = " ".join(stream_parts).encode("latin1")
            stream = DecodedStreamObject()
            stream.set_data(stream_data)
            page[NameObject("/Contents")] = stream

    with open(file_path, "wb") as f:
        writer.write(f)
    return file_path


def test_pdf_loader_multi_page_and_empty_pages(tmp_path: Path) -> None:
    """Test PDF extraction with multiple pages, blank pages, and content hashing."""
    pdf_path = tmp_path / "sample.pdf"
    create_test_pdf(
        pdf_path,
        [
            "Page 1: Introduction to GraphRAG architectures.\nKey mechanisms explored.",
            None,  # Blank page (Page 2)
            "Page 3: Evaluation results and conclusions.",
        ],
    )

    loader = PDFLoader()
    doc, pages = loader.load_file(pdf_path, kb_id="test_kb", doc_id="D001")

    assert doc.doc_id == "D001"
    assert doc.kb_id == "test_kb"
    assert doc.filename == "sample.pdf"
    assert doc.total_pages == 3
    assert doc.empty_pages == [2]
    assert doc.is_scanned_or_empty is False
    assert len(doc.content_hash) == 64

    assert len(pages) == 3
    assert pages[0].page_number == 1
    assert "GraphRAG architectures" in pages[0].text
    assert pages[0].is_empty is False
    assert pages[0].char_count == len(pages[0].text)
    assert pages[0].content_hash == hashlib.sha256(pages[0].text.encode("utf-8")).hexdigest()

    assert pages[1].page_number == 2
    assert pages[1].text == ""
    assert pages[1].is_empty is True
    assert pages[1].char_count == 0

    assert pages[2].page_number == 3
    assert "Evaluation results" in pages[2].text
    assert pages[2].is_empty is False


def test_pdf_loader_scanned_or_empty_detection(tmp_path: Path) -> None:
    """Test detection of scanned or empty documents where all pages contain no text."""
    scanned_pdf = tmp_path / "scanned.pdf"
    create_test_pdf(scanned_pdf, [None, None])

    loader = PDFLoader()
    doc, pages = loader.load_file(scanned_pdf, kb_id="test_kb", doc_id="D002")

    assert doc.total_pages == 2
    assert doc.empty_pages == [1, 2]
    assert doc.is_scanned_or_empty is True
    for p in pages:
        assert p.is_empty is True

    # Also test an empty .txt file
    empty_txt = tmp_path / "empty.txt"
    empty_txt.write_text("   \n\n  ", encoding="utf-8")
    doc_txt, pages_txt = loader.load_file(empty_txt, kb_id="test_kb", doc_id="D003")
    assert doc_txt.total_pages == 1
    assert doc_txt.empty_pages == [1]
    assert doc_txt.is_scanned_or_empty is True
    assert pages_txt[0].is_empty is True


def test_pdf_loader_deterministic_sorting_and_ids(tmp_path: Path) -> None:
    """Test directory loading sorts filenames deterministically and assigns sequential IDs."""
    data_dir = tmp_path / "raw_docs"
    data_dir.mkdir()

    # Create files out of alphabetical order
    (data_dir / "charlie.md").write_text("# Charlie Doc\nSome markdown text.", encoding="utf-8")
    (data_dir / "alpha.txt").write_text("Alpha document content.", encoding="utf-8")
    create_test_pdf(data_dir / "bravo.pdf", ["Bravo PDF content."])

    loader = PDFLoader()
    docs, pages = loader.load_directory(data_dir, kb_id="academic_kb")

    assert len(docs) == 3
    # Check alphabetical ordering and ID assignment
    assert docs[0].doc_id == "D001"
    assert docs[0].filename == "alpha.txt"
    assert docs[0].kb_id == "academic_kb"

    assert docs[1].doc_id == "D002"
    assert docs[1].filename == "bravo.pdf"
    assert docs[1].kb_id == "academic_kb"

    assert docs[2].doc_id == "D003"
    assert docs[2].filename == "charlie.md"
    assert docs[2].kb_id == "academic_kb"


def test_pdf_loader_synthetic_text_form_feed(tmp_path: Path) -> None:
    """Test multi-page synthetic text files separated by form feed '\\f'."""
    text_file = tmp_path / "synthetic_multipage.txt"
    text_file.write_text(
        "Page 1 text on hybrid graph retrieval.\f   \r\n\t  \fPage 3 text with evaluation metrics.",
        encoding="utf-8",
    )

    loader = PDFLoader()
    doc, pages = loader.load_file(text_file, kb_id="test_kb", doc_id="D001")

    assert doc.total_pages == 3
    assert doc.empty_pages == [2]
    assert doc.is_scanned_or_empty is False
    assert len(pages) == 3
    assert pages[0].page_number == 1
    assert "hybrid graph retrieval" in pages[0].text
    assert pages[1].page_number == 2
    assert pages[1].is_empty is True
    assert pages[2].page_number == 3
    assert "evaluation metrics" in pages[2].text


def test_chunker_cross_page_provenance_and_deterministic_ids() -> None:
    """Test chunking with cross-page provenance ([1, 2]) and deterministic chunk IDs."""
    kb_id = "test_kb"
    doc_id = "D001"

    # Page 1: 600 chars, Page 2: 700 chars
    p1_text = "Page 1 start. " + ("Important research fact. " * 23)
    p2_text = "Page 2 continuation. " + ("Additional findings on graphrag. " * 20)

    p1 = Page(
        kb_id=kb_id,
        doc_id=doc_id,
        filename="report.pdf",
        page_number=1,
        text=p1_text,
        char_count=len(p1_text),
        is_empty=False,
        content_hash=hashlib.sha256(p1_text.encode("utf-8")).hexdigest(),
    )
    p2 = Page(
        kb_id=kb_id,
        doc_id=doc_id,
        filename="report.pdf",
        page_number=2,
        text=p2_text,
        char_count=len(p2_text),
        is_empty=False,
        content_hash=hashlib.sha256(p2_text.encode("utf-8")).hexdigest(),
    )

    config = ChunkingConfig(chunk_size=1100, chunk_overlap=180, respect_sentence_boundaries=True)
    chunker = DeterministicChunker(config=config)
    chunks = chunker.chunk_pages([p1, p2], kb_id=kb_id)

    assert len(chunks) >= 2
    # First chunk spans across page 1 and page 2
    assert chunks[0].chunk_id == "D001_C0000"
    assert chunks[0].doc_id == "D001"
    assert chunks[0].chunk_index == 0
    assert chunks[0].page_numbers == [1, 2]
    assert len(chunks[0].content_hash) == 64
    assert chunks[0].char_count == len(chunks[0].text)

    # Second chunk continues deterministically
    assert chunks[1].chunk_id == "D001_C0001"
    assert chunks[1].chunk_index == 1
    assert 2 in chunks[1].page_numbers


def test_chunker_boundary_preference_hierarchy() -> None:
    """Test boundary hierarchy preference: paragraph -> newline -> sentence -> hard cut."""
    kb_id = "test_kb"
    config = ChunkingConfig(chunk_size=1100, chunk_overlap=180, respect_sentence_boundaries=True)
    chunker = DeterministicChunker(config=config)

    # 1. Paragraph boundary preference: \n\n in last 25% (between 825 and 1100)
    text_para = ("A" * 950) + "\n\n" + ("B" * 300)
    p_para = Page(
        kb_id=kb_id,
        doc_id="D001",
        filename="p.txt",
        page_number=1,
        text=text_para,
        char_count=len(text_para),
        is_empty=False,
        content_hash="h1",
    )
    chunks_para = chunker.chunk_pages([p_para], kb_id=kb_id)
    # The first chunk should break right at the paragraph break (length 950)
    assert chunks_para[0].text == "A" * 950
    assert "B" not in chunks_para[0].text

    # 2. Single newline boundary preference: \n when \n\n is absent
    text_nl = ("C" * 950) + "\n" + ("D" * 300)
    p_nl = Page(
        kb_id=kb_id,
        doc_id="D002",
        filename="nl.txt",
        page_number=1,
        text=text_nl,
        char_count=len(text_nl),
        is_empty=False,
        content_hash="h2",
    )
    chunks_nl = chunker.chunk_pages([p_nl], kb_id=kb_id)
    assert chunks_nl[0].text == "C" * 950
    assert "D" not in chunks_nl[0].text

    # 3. Sentence boundary preference: '. ' when no newlines
    text_sent = ("E" * 950) + ". " + ("F" * 300)
    p_sent = Page(
        kb_id=kb_id,
        doc_id="D003",
        filename="sent.txt",
        page_number=1,
        text=text_sent,
        char_count=len(text_sent),
        is_empty=False,
        content_hash="h3",
    )
    chunks_sent = chunker.chunk_pages([p_sent], kb_id=kb_id)
    assert chunks_sent[0].text == ("E" * 950) + "."
    assert "F" not in chunks_sent[0].text

    # 4. Hard cut fallback: no boundaries at all
    text_hard = "G" * 1500
    p_hard = Page(
        kb_id=kb_id,
        doc_id="D004",
        filename="hard.txt",
        page_number=1,
        text=text_hard,
        char_count=len(text_hard),
        is_empty=False,
        content_hash="h4",
    )
    chunks_hard = chunker.chunk_pages([p_hard], kb_id=kb_id)
    assert len(chunks_hard[0].text) == 1100
    assert chunks_hard[0].text == "G" * 1100


def test_chunker_skips_empty_pages() -> None:
    """Test that chunker skips empty pages and maintains clean provenance."""
    kb_id = "test_kb"
    doc_id = "D001"
    p1 = Page(
        kb_id=kb_id,
        doc_id=doc_id,
        filename="doc.pdf",
        page_number=1,
        text="Content of page 1.",
        char_count=18,
        is_empty=False,
        content_hash="h1",
    )
    p2 = Page(
        kb_id=kb_id,
        doc_id=doc_id,
        filename="doc.pdf",
        page_number=2,
        text="",
        char_count=0,
        is_empty=True,
        content_hash="h2",
    )
    p3 = Page(
        kb_id=kb_id,
        doc_id=doc_id,
        filename="doc.pdf",
        page_number=3,
        text="Content of page 3.",
        char_count=18,
        is_empty=False,
        content_hash="h3",
    )

    chunker = DeterministicChunker()
    chunks = chunker.chunk_pages([p1, p2, p3], kb_id=kb_id)
    assert len(chunks) == 1
    # Spans page 1 and page 3, skipping empty page 2
    assert chunks[0].page_numbers == [1, 3]


def test_document_store_roundtrip_and_lookups(tmp_path: Path) -> None:
    """Test DocumentStore JSONL persistence, reload, and O(1) chunk lookups."""
    kb_id = "academic_kb"
    store = DocumentStore(kb_id=kb_id, base_dir=tmp_path)

    doc = Document(
        kb_id=kb_id,
        doc_id="D001",
        filename="paper.pdf",
        source_path="/tmp/paper.pdf",
        content_hash="doc_hash_123",
        total_pages=2,
        empty_pages=[],
        is_scanned_or_empty=False,
        metadata={"domain": "cs"},
    )
    page1 = Page(
        kb_id=kb_id,
        doc_id="D001",
        filename="paper.pdf",
        page_number=1,
        text="Page 1 text.",
        char_count=12,
        is_empty=False,
        content_hash="p1_hash",
    )
    page2 = Page(
        kb_id=kb_id,
        doc_id="D001",
        filename="paper.pdf",
        page_number=2,
        text="Page 2 text.",
        char_count=12,
        is_empty=False,
        content_hash="p2_hash",
    )
    chunk0 = Chunk(
        kb_id=kb_id,
        chunk_id="D001_C0000",
        doc_id="D001",
        filename="paper.pdf",
        page_numbers=[1],
        chunk_index=0,
        text="Page 1 text.",
        char_count=12,
        content_hash="c0_hash",
        metadata={"start_char": 0, "end_char": 12},
    )
    chunk1 = Chunk(
        kb_id=kb_id,
        chunk_id="D001_C0001",
        doc_id="D001",
        filename="paper.pdf",
        page_numbers=[2],
        chunk_index=1,
        text="Page 2 text.",
        char_count=12,
        content_hash="c1_hash",
        metadata={"start_char": 14, "end_char": 26},
    )

    store.save_all([doc], [page1, page2], [chunk0, chunk1])

    # Verify JSONL files exist
    assert store.documents_path.exists()
    assert store.pages_path.exists()
    assert store.chunks_path.exists()

    # Fast in-memory lookups
    c0 = store.get_chunk("D001_C0000")
    assert c0 is not None
    assert c0.chunk_id == "D001_C0000"
    assert c0.text == "Page 1 text."

    assert store.get_chunk("NON_EXISTENT") is None

    # Multi-chunk lookup preserving order
    batch = store.get_chunks_by_ids(["D001_C0001", "D001_C0000", "MISSING"])
    assert len(batch) == 2
    assert batch[0].chunk_id == "D001_C0001"
    assert batch[1].chunk_id == "D001_C0000"

    all_chunks = store.list_all_chunks()
    assert len(all_chunks) == 2

    # Verify reload from disk with a fresh DocumentStore instance
    store2 = DocumentStore(kb_id=kb_id, base_dir=tmp_path)
    loaded_docs = store2.load_documents()
    loaded_pages = store2.load_pages()
    loaded_chunks = store2.load_chunks()

    assert len(loaded_docs) == 1
    assert loaded_docs[0] == doc
    assert len(loaded_pages) == 2
    assert loaded_pages == [page1, page2]
    assert len(loaded_chunks) == 2
    assert loaded_chunks == [chunk0, chunk1]

    # Fast lookup works immediately after cold reload
    assert store2.get_chunk("D001_C0000") == chunk0


def test_document_store_kb_isolation_enforcement(tmp_path: Path) -> None:
    """Test that DocumentStore rejects and filters records with mismatched kb_id."""
    store = DocumentStore(kb_id="kb_alpha", base_dir=tmp_path)

    invalid_chunk = Chunk(
        kb_id="kb_beta",
        chunk_id="D001_C0000",
        doc_id="D001",
        filename="paper.pdf",
        page_numbers=[1],
        chunk_index=0,
        text="Cross-KB leak test.",
        char_count=19,
        content_hash="hash",
    )

    with pytest.raises(ValueError, match="does not match store kb_id"):
        store.save_chunks([invalid_chunk])

    invalid_doc = Document(
        kb_id="kb_beta",
        doc_id="D001",
        filename="paper.pdf",
        source_path="/tmp/paper.pdf",
        content_hash="doc_hash",
        total_pages=1,
    )

    with pytest.raises(ValueError, match="does not match store kb_id"):
        store.save_documents([invalid_doc])


def test_chunker_multi_document_and_punctuation_varieties() -> None:
    """Test chunking multiple documents with per-document ID reset and question/exclamation marks."""
    kb_id = "test_kb"
    config = ChunkingConfig(chunk_size=1100, chunk_overlap=180, respect_sentence_boundaries=True)
    chunker = DeterministicChunker(config=config)

    # Document 1 with '?' sentence boundary
    text_q = ("Question context " * 50) + "What is the result? " + ("Followup text " * 30)
    p_d1 = Page(
        kb_id=kb_id,
        doc_id="D001",
        filename="doc1.txt",
        page_number=1,
        text=text_q,
        char_count=len(text_q),
        is_empty=False,
        content_hash="h_d1",
    )

    # Document 2 with '!' sentence boundary
    # 'Alert context ' is 14 chars. 60 * 14 = 840 chars.
    # 'System is operational! ' is 23 chars. Total 863 chars (in 825..1100 window).
    text_ex = ("Alert context " * 60) + "System is operational! " + ("Status log " * 50)
    p_d2 = Page(
        kb_id=kb_id,
        doc_id="D002",
        filename="doc2.txt",
        page_number=1,
        text=text_ex,
        char_count=len(text_ex),
        is_empty=False,
        content_hash="h_d2",
    )

    chunks = chunker.chunk_pages([p_d1, p_d2], kb_id=kb_id)

    # Filter by doc_id
    d1_chunks = [c for c in chunks if c.doc_id == "D001"]
    d2_chunks = [c for c in chunks if c.doc_id == "D002"]

    assert len(d1_chunks) >= 2
    assert d1_chunks[0].chunk_id == "D001_C0000"
    assert d1_chunks[0].chunk_index == 0
    assert d1_chunks[0].text.endswith("What is the result?")

    assert len(d2_chunks) >= 2
    assert d2_chunks[0].chunk_id == "D002_C0000"
    assert d2_chunks[0].chunk_index == 0
    assert d2_chunks[0].text.endswith("System is operational!")


def test_document_store_doc_and_page_lookups(tmp_path: Path) -> None:
    """Test DocumentStore get_document, list_all_documents, and get_pages_for_document."""
    kb_id = "lookup_kb"
    store = DocumentStore(kb_id=kb_id, base_dir=tmp_path)

    doc1 = Document(
        kb_id=kb_id,
        doc_id="D001",
        filename="d1.pdf",
        source_path="/tmp/d1.pdf",
        content_hash="hash1",
        total_pages=2,
    )
    doc2 = Document(
        kb_id=kb_id,
        doc_id="D002",
        filename="d2.pdf",
        source_path="/tmp/d2.pdf",
        content_hash="hash2",
        total_pages=1,
    )

    p1_d1 = Page(
        kb_id=kb_id,
        doc_id="D001",
        filename="d1.pdf",
        page_number=1,
        text="Page 1",
        char_count=6,
        is_empty=False,
        content_hash="ph1",
    )
    p2_d1 = Page(
        kb_id=kb_id,
        doc_id="D001",
        filename="d1.pdf",
        page_number=2,
        text="Page 2",
        char_count=6,
        is_empty=False,
        content_hash="ph2",
    )

    store.save_documents([doc1, doc2])
    store.save_pages([p2_d1, p1_d1])  # saved out-of-order to test sorting

    assert store.get_document("D001") == doc1
    assert store.get_document("D002") == doc2
    assert store.get_document("D999") is None

    all_docs = store.list_all_documents()
    assert len(all_docs) == 2

    d1_pages = store.get_pages_for_document("D001")
    assert len(d1_pages) == 2
    assert d1_pages[0].page_number == 1
    assert d1_pages[1].page_number == 2
