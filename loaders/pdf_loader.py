from pathlib import Path

import pymupdf


def load_file(path: str | Path) -> list[dict]:
    """Extract one record per non-empty page, with paragraphs joined by \\n\\n.

    PyMuPDF text blocks correspond closely to visual paragraphs, so block
    texts are re-joined with blank lines to give the chunker a clean
    paragraph-delimited page text.
    """
    path = Path(path)
    records = []
    with pymupdf.open(path) as doc:
        for page_num in range(doc.page_count):
            blocks = doc[page_num].get_text("blocks")
            paragraphs = []
            for block in blocks:
                if len(block) < 7 or block[6] != 0:
                    continue
                text = block[4]
                if not isinstance(text, str):
                    continue
                text = text.strip()
                if text:
                    paragraphs.append(text)
            if paragraphs:
                records.append(
                    {
                        "source": path.name,
                        "page": page_num + 1,
                        "text": "\n\n".join(paragraphs),
                    }
                )
    return records
