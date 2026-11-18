"""Document loading implementation supporting PDF, TXT, and Markdown files."""

from __future__ import annotations
import hashlib
import logging
from pathlib import Path
from typing import List, Tuple, Union
import pypdf
from src.core.interfaces import DocumentLoader
from src.core.models import Document, Page

logger = logging.getLogger(__name__)


class PDFLoader(DocumentLoader):
    """Loads documents from a directory, supporting PDF, TXT, and Markdown files.

    Maintains deterministic document IDs (D001, D002, ...) and page-level provenance.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

    def __init__(self) -> None:
        pass

    def load_directory(
        self, input_dir: Union[str, Path], kb_id: str
    ) -> Tuple[List[Document], List[Page]]:
        """Load all supported documents in input_dir sorted deterministically.

        Assigns sequential doc_ids (D001, D002, ...) and extracts individual pages.
        """
        dir_path = Path(input_dir)
        if not dir_path.exists():
            raise FileNotFoundError(f"Input directory does not exist: {dir_path}")
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {dir_path}")

        candidate_files = [
            p
            for p in dir_path.iterdir()
            if p.is_file()
            and p.suffix.lower() in self.SUPPORTED_EXTENSIONS
            and not p.name.startswith(".")
        ]
        # Sort files deterministically by filename
        candidate_files.sort(key=lambda p: p.name)

        all_documents: List[Document] = []
        all_pages: List[Page] = []

        for doc_index, file_path in enumerate(candidate_files, start=1):
            doc_id = f"D{doc_index:03d}"
            doc, pages = self.load_file(file_path=file_path, kb_id=kb_id, doc_id=doc_id)
            all_documents.append(doc)
            all_pages.extend(pages)

        return all_documents, all_pages

    def load_file(
        self, file_path: Union[str, Path], kb_id: str, doc_id: str
    ) -> Tuple[Document, List[Page]]:
        """Load a single document file, returning the Document and its constituent Pages."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File does not exist: {path}")

        file_bytes = path.read_bytes()
        doc_content_hash = hashlib.sha256(file_bytes).hexdigest()
        ext = path.suffix.lower()

        raw_pages_text: List[str] = []
        if ext == ".pdf":
            raw_pages_text = self._extract_pdf_pages(path)
        elif ext in {".txt", ".md"}:
            raw_pages_text = self._extract_text_pages(path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        pages: List[Page] = []
        empty_pages: List[int] = []

        # If file produced no pages at all (e.g. empty file), handle gracefully as 1 empty page
        if not raw_pages_text:
            raw_pages_text = [""]

        for page_num, raw_text in enumerate(raw_pages_text, start=1):
            normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n")
            lines = [line.rstrip() for line in normalized.split("\n")]
            cleaned_text = "\n".join(lines).rstrip()

            is_empty = len(cleaned_text.strip()) == 0
            if is_empty:
                cleaned_text = ""
                empty_pages.append(page_num)

            page_hash = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
            page = Page(
                kb_id=kb_id,
                doc_id=doc_id,
                filename=path.name,
                page_number=page_num,
                text=cleaned_text,
                char_count=len(cleaned_text),
                is_empty=is_empty,
                content_hash=page_hash,
            )
            pages.append(page)

        total_pages = len(pages)
        is_scanned_or_empty = (len(empty_pages) == total_pages and total_pages > 0) or total_pages == 0

        if is_scanned_or_empty:
            logger.warning(
                "Document %s (%s) has no extractable text across all %d pages (may be scanned or empty).",
                doc_id,
                path.name,
                total_pages,
            )

        document = Document(
            kb_id=kb_id,
            doc_id=doc_id,
            filename=path.name,
            source_path=str(path.resolve()),
            content_hash=doc_content_hash,
            total_pages=total_pages,
            empty_pages=empty_pages,
            is_scanned_or_empty=is_scanned_or_empty,
            metadata={
                "extension": ext,
                "file_size": len(file_bytes),
            },
        )

        return document, pages

    def _extract_pdf_pages(self, file_path: Path) -> List[str]:
        """Extract text from each page of a PDF using pypdf."""
        reader = pypdf.PdfReader(str(file_path))
        extracted: List[str] = []
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
            except Exception as e:
                logger.warning(
                    "Error extracting text from page in PDF %s: %s", file_path.name, e
                )
                text = ""
            extracted.append(text)
        return extracted

    def _extract_text_pages(self, file_path: Path) -> List[str]:
        """Extract text pages from .txt or .md files, splitting on form-feed if present."""
        raw_content = file_path.read_text(encoding="utf-8", errors="replace")
        if "\f" in raw_content:
            return raw_content.split("\f")
        return [raw_content]
