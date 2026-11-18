#!/usr/bin/env python3
"""Embed documents, answerable questions and B7 second-stage queries for a synthetic corpus.

Same procedure as experiments/synth_kg_audit/audit_part1_a.py (documents: stripped full text; questions: raw text;
OllamaEmbeddingService with default OllamaConfig, batches of 25) and the B7 definition in
experiments/synth_kg_audit/followup/PROTOCOL.md (q2 = question + " " + text(d1), d1 = dense top-1 document,
text(d1) truncated to its first 2,000 characters). The stored seed-7 q2 vectors are reproduced exactly
(cosine 1.0) only when text(d1) is the RAW file text (not stripped), so the raw text is used for q2.
Embedding calls only.

Usage: python3 experiments/synth_kg_fix/embed_corpus.py <corpus dir> <out.npz>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from src.core.configuration import OllamaConfig  # noqa: E402
from src.embeddings.embedding_service import OllamaEmbeddingService  # noqa: E402


def embed(service, texts, batch=25):
    out = []
    for i in range(0, len(texts), batch):
        out.extend(service.embed_batch(texts[i:i + batch]))
    return np.array(out, dtype=np.float32)


def main() -> None:
    corpus, out = Path(sys.argv[1]), Path(sys.argv[2])
    doc_files = sorted((corpus / "docs").glob("*.txt"), key=lambda p: p.name)
    doc_ids = [p.stem for p in doc_files]
    raw_texts = [p.read_text(encoding="utf-8") for p in doc_files]
    doc_texts = [t.strip() for t in raw_texts]
    questions = [json.loads(l) for l in (corpus / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    questions = [q for q in questions if q.get("group") != "unanswerable"]
    service = OllamaEmbeddingService(config=OllamaConfig())
    doc_mat = embed(service, doc_texts)
    q_mat = embed(service, [q["question"] for q in questions])
    q2_texts = []
    for i, q in enumerate(questions):
        d1 = int(np.argmax(np.dot(doc_mat, q_mat[i])))
        q2_texts.append(q["question"] + " " + raw_texts[d1][:2000])
    q2_mat = embed(service, q2_texts)
    np.savez(out, doc_mat=doc_mat, doc_ids=np.array(doc_ids), q_mat=q_mat, q2_mat=q2_mat,
             q_ids=np.array([q["id"] for q in questions]))
    print(f"saved {out}: docs={doc_mat.shape} questions={q_mat.shape} q2={q2_mat.shape}")


if __name__ == "__main__":
    main()
