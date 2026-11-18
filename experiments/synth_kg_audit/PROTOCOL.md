# Pre-Registration Protocol: Graph-Free Retrieval Baselines

**Date**: 2026-11-16  
**Evaluation Scope**: 105 answerable questions of `experiments/synth_kg/corpus_final/questions.jsonl` (30 hop1, 30 hop2, 30 hop3, 15 alias).  
**Corpus**: `experiments/synth_kg/corpus_final/docs/` (60 documents: `DOC_0001.txt` through `DOC_0060.txt`).  
**Retrieval Budget**: $K = 4$ documents (budget-matched to engine `final_context_chunks=4` / `vector_top_k=4`), plus non-budget reference B6 ($K = 8$).  
**Metric**: Full Gold Evidence Recall — fraction of questions where **all** gold supporting documents (`gold_supporting_doc_ids`) are contained within the top-$K$ returned documents. Evaluated per question group with Wilson 95% score confidence intervals.

---

## 1. Baseline Definitions

1. **B1: Dense Top-4**
   - Query and document embeddings produced by `embeddinggemma:latest` (cosine similarity).
   - Return top-4 documents sorted by cosine similarity.

2. **B2: BM25 Top-4**
   - Lexical baseline implemented directly without external libraries.
   - Tokenization: regex `\b[a-zA-Z0-9]+\b`, lowercased.
   - Parameters: $k_1 = 1.5$, $b = 0.75$.
   - Standard BM25 IDF: $\ln \left( \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1 \right)$.
   - Return top-4 documents by BM25 score.

3. **B3: Reciprocal Rank Fusion (RRF Top-4)**
   - RRF combination of B1 (Dense) and B2 (BM25) rankings.
   - Constant $k = 60$.
   - Score for document $d$: $RRF(d) = \frac{1}{60 + \text{rank}_{dense}(d)} + \frac{1}{60 + \text{rank}_{bm25}(d)}$.
   - Return top-4 documents sorted by descending RRF score.

4. **B4: Name Expansion Top-4**
   - Seed documents: top-2 documents from B1 (Dense).
   - Name extraction: Extract all maximal runs of capitalized words from the text of the 2 seed documents (regex `\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b`). No access to `gold_kg.json`.
   - Candidate pool: All corpus documents (excluding seeds) whose raw text contains at least one extracted name.
   - Candidate ranking: Rank candidate documents by cosine similarity to the question using the dense query embedding.
   - Final set: The 2 seed documents + top-2 ranked candidate documents. If fewer than 2 candidates exist, backfill remaining slots from the top of the dense ranking (B1) excluding already selected documents.

5. **B6: Dense Top-8 (Reference)**
   - Top-8 documents from B1 (Dense). Non-budget matched reference.

---

## 2. Fixed Decision Rule

Let $H$ be the hybrid mode's `hop2` all-gold evidence recall observed in `experiments/synth_kg/runs` (56.7% in `REPORT.md`; recomputed from run records).
Compare $B^* = \max(\text{Recall}_{B2}, \text{Recall}_{B3}, \text{Recall}_{B4})$ on `hop2`:
- If $B^* \ge H - 10.0$ percentage points: Conclude **"graph-free methods reproduce most of the evidence-recall gain; graph-specific benefit not demonstrated"**.
- If $B^* < H - 20.0$ percentage points: Conclude **"graph-specific benefit supported on this corpus"**.
- Otherwise: Conclude **"ambiguous"**.

*Note*: The decision rule applies strictly to the `hop2` slice. Performance on `hop3` and `alias` will be reported for completeness but does not alter the decision rule.
