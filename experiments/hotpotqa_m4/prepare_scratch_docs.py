#!/usr/bin/env python3
"""Part 3 Preparation: Write the 10 context paragraphs of the first bridge question to scratch_docs."""

from __future__ import annotations

import json
from pathlib import Path


def safe_filename(title: str, max_len: int = 50) -> str:
    clean = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in title)
    return clean[:max_len]


def main() -> None:
    questions_file = Path("experiments/hotpotqa_m3/selected_questions.json")
    scratch_docs_dir = Path("experiments/hotpotqa_m4/scratch_docs")
    scratch_docs_dir.mkdir(parents=True, exist_ok=True)

    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # First bridge question
    q0 = next(q for q in questions if q.get("type") == "bridge")
    print(f"First bridge question: {q0['_id']}")
    print(f"Question text: {q0['question']}")
    print(f"Context paragraphs count: {len(q0['context'])}")

    written_files = []
    for idx, (title, sentences) in enumerate(q0["context"], start=1):
        clean_title = safe_filename(title)
        fname = f"doc_{idx:02d}_{clean_title}.txt"
        fpath = scratch_docs_dir / fname

        joined_sentences = " ".join(s.strip() for s in sentences if s.strip())
        content = f"{title}\n{joined_sentences}\n"

        fpath.write_text(content, encoding="utf-8")
        written_files.append((fname, title, len(sentences), len(content)))
        print(f"[{idx:02d}/10] Wrote {fname} (Title: '{title}', {len(sentences)} sents, {len(content)} chars)")

    print(f"\nSuccessfully wrote {len(written_files)} documents to {scratch_docs_dir}")


if __name__ == "__main__":
    main()
