# Pre-Registration Protocol: Stronger Graph-Free Iterative Baselines

**Date**: 2026-11-16  
**Evaluation Scope**: 105 answerable questions of `experiments/synth_kg/corpus_final/questions.jsonl` (primary focus: 30 hop2 questions; 30 hop1, 30 hop3, 15 alias reported for completeness).  
**Corpus**: `experiments/synth_kg/corpus_final/docs/` (60 documents: `DOC_0001.txt` through `DOC_0060.txt`).  
**Retrieval Budget**: $K = 4$ documents (strictly budget-matched to engine `final_context_chunks=4` / `vector_top_k=4`).  
**Metric**: Full Gold Evidence Recall — fraction of questions where **all** gold supporting documents (`gold_supporting_doc_ids`) are contained within the top-$K$ returned documents. Evaluated per question group with Wilson 95% score confidence intervals.  
**Embedding Model**: `embeddinggemma:latest` (768-dim, Ollama).

---

## 1. Baseline Definitions

1. **B7: Iterative Dense Retrieval (Top-4)**
   - Query question $q$ embedded with `embeddinggemma:latest`.
   - $d_1$: dense top-1 document for $q$ by cosine similarity.
   - Second-stage expanded query: $q_2 = q + \text{" "} + \text{text}(d_1)$.
   - Truncation rule: If text of $d_1$ exceeds 2,000 characters, truncate to the first 2,000 characters to fit within embedding context bounds safely without truncation errors.
   - Embed $q_2$ using `embeddinggemma:latest`.
   - Rank all other corpus documents ($D \setminus \{d_1\}$) by cosine similarity to $q_2$.
   - Final retrieved set: $d_1$ plus top-3 ranked documents from the second stage (total 4 documents).

2. **B8: Overlap Expansion (Top-2 Seeds + Top-2 Overlap Candidates)**
   - Seeds: dense top-2 documents ($s_1, s_2$) for $q$.
   - Name extraction: Extract names from text of seeds as maximal runs of capitalized words using regex `\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b` (no access to `gold_kg.json`).
   - Question filtering: Remove any extracted name that also appears (case-insensitive substring) in the question $q$.
   - Candidate pool: All other documents ($D \setminus \{s_1, s_2\}$) that contain at least one remaining extracted name.
   - Candidate ranking: Rank candidate documents by the number of distinct remaining names they contain (descending). Break ties by cosine similarity to the question $q$ using the dense embedding.
   - Final set: The 2 seeds ($s_1, s_2$) plus top-2 ranked candidates. If fewer than 2 candidates exist, backfill remaining slots from the dense ranking (B1) excluding already selected documents.

3. **B9: Overlap Expansion (Top-1 Seed + Top-3 Overlap Candidates)**
   - Identical to B8, but seeds = dense top-1 document ($s_1$).
   - Returns $s_1$ plus top-3 ranked candidates (backfilling from dense ranking if fewer than 3 candidates exist).

4. **B4: Name Expansion Top-4 (Reference from Previous Audit)**
   - Seeds = dense top-2 documents.
   - Extract capitalized names from seeds.
   - Candidate documents = all other documents containing at least one extracted name.
   - Rank candidates strictly by cosine similarity to question $q$.
   - Final set: 2 seeds + top-2 candidates (backfill from dense if needed).

---

## 2. Fixed Decision Rule (Evaluated on hop2 Slice Only)

Let $H$ be the hybrid mode's `hop2` all-gold evidence recall observed in `experiments/synth_kg/runs` (56.7% in `REPORT.md`; recomputed from run records).
Let $B^* = \max(\text{Recall}_{B7}, \text{Recall}_{B8}, \text{Recall}_{B9})$ on `hop2`.

- If $B^* \ge H - 10.0$ percentage points ($B^* \ge 46.7\%$):  
  Conclude: **"iterative retrieval without a graph reproduces most of the hybrid gain; graph-specific benefit not demonstrated"**.
- If $B^* < H - 20.0$ percentage points ($B^* < 36.7\%$):  
  Conclude: **"graph-specific benefit supported against iterative baselines"**.
- Otherwise ($36.7\% \le B^* < 46.7\%$):  
  Conclude: **"ambiguous"**.

*Descriptive Analysis*: For each of B4, B7, B8, B9, report paired comparison against hybrid on the 30 hop2 questions ($n_{11}, n_{10}, n_{01}, n_{00}$) and exact two-sided sign-test $p$-value.
