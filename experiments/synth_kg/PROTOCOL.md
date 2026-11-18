# Pre-Registration Protocol: Synthetic KG Benchmark

**Protocol Version**: 1.0  
**Date**: 2026-11-16  
**Corpus**: `experiments/synth_kg/corpus_final/`  
**Ontology**: `configs/domains/company.yaml` merged with `configs/default.yaml`  

---

## 1. System Configuration & Query Settings

All experimental runs use the public API `hybrid_graphrag.HybridGraphRAG`.

- **Base Configuration**: `configs/default.yaml` merged with `configs/domains/company.yaml`.
- **Knowledge Base Storage Root**: `experiments/synth_kg/kb/` (KB ID: `synth_company_kb`).
- **LLM Model**: `gemma3:4b` via Ollama at `http://localhost:11434`.
- **LLM Temperature**: `0.0`.
- **Embedding Model**: `embeddinggemma:latest` (or configured embedding client in default config).
- **Retrieval Parameters**:
  - `vector_top_k`: `4`
  - `final_context_chunks`: `4`
  - `comparative_reasoning_mode`: `"off"`
  - `graph_context_mode`: `"off"`
  - `prompt_version`: `"v1.1"` set explicitly
- **All other parameters**: Default configuration as specified in `configs/default.yaml` and domain overrides.
- **Snapshot**: An effective runtime configuration snapshot is persisted into each run directory alongside results.

---

## 2. Execution Order & Priority

Execution proceeds strictly in the following priority order:
1. **Build Knowledge Base**: Single ingestion of `experiments/synth_kg/corpus_final/docs/` through `HybridGraphRAG.ingest()`.
2. **Evaluation Runs** (each runs the 120 questions in `experiments/synth_kg/corpus_final/questions.jsonl` in a fresh process):
   - Run 1: `vector_only`
   - Run 2: `hybrid`
   - Run 3: `graph_only`
   - Run 4: `community_aware`
3. **Offline Scoring**: Execution of `score.py` over results and `gold_kg.json`.
4. **Report Generation**: Execution of `report.py` producing `REPORT.md`.

---

## 3. Answer Correctness & Evaluation Rules

### Text Normalization
Predictions and gold answers/aliases are normalized by:
1. Converting to lower case.
2. Removing punctuation characters (retaining alphanumeric and whitespace).
3. Collapsing consecutive whitespace into a single space and stripping leading/trailing whitespace.

### Correctness Criteria
- **Answerable and Alias Groups (`hop1`, `hop2`, `hop3`, `alias`)**:
  - The normalized prediction contains the normalized gold canonical name OR any gold alias as a **whole-word phrase** (matching regex `r"\b" + re.escape(cand) + r"\b"`).
  - AND `abstained is False`.
  - AND `generation_error is False`.
- **Unanswerable Group (`unanswerable`)**:
  - Correct IF AND ONLY IF `abstained is True` AND `generation_error is False`.
- **Generation Errors (`generation_error is True`)**:
  - Counted as `ERROR` in a distinct column.
  - Never counted as correct, and never counted as an abstention.

### Failure Categorization (for incorrect answers in answerable/alias groups)
1. **ERROR**: `generation_error is True`.
2. **ABSTAINED**: `generation_error is False` and `abstained is True`.
3. **WRONG_ENTITY**: Prediction does not contain the correct entity, but contains another entity name (canonical or alias) from `gold_kg.json` belonging to the **same entity type** as the gold answer.
4. **OTHER**: Any other incorrect response (e.g., hallucinated entity not in KG, vague text, wrong type).

---

## 4. Evidence Retrieval Metric

For each question:
- Context chunk IDs returned in the result record are mapped back to their originating source document IDs.
- A question achieves **full evidence recall** if and only if **all** document IDs in `gold_supporting_doc_ids` are present among the retrieved context documents.
- Evaluated as the fraction of questions achieving full evidence recall, broken down by question group.

---

## 5. Statistical Hypothesis Testing & Comparisons

### Primary Comparisons
- `hybrid` vs `vector_only`
- `graph_only` vs `vector_only`
- `community_aware` vs `vector_only`

### Metric & Test
- Evaluated on accuracy over all answerable questions, and broken down per hop group (`hop1`, `hop2`, `hop3`, `alias`).
- Paired contingency counts on discordant pairs:
  - $n_{10}$: Correct in mode A, incorrect in mode B.
  - $n_{01}$: Incorrect in mode A, correct in mode B.
- **Statistical Test**: Exact two-sided sign test (equivalent to exact binomial test on discordant pairs with null probability $p = 0.5$):
  $$\text{discordant} = n_{10} + n_{01}$$
  $$p\text{-value} = \text{binom\_test}(\min(n_{10}, n_{01}), n_{10} + n_{01}, 0.5)$$
- **Significance Criterion**: A difference is claimed as "detected" if and only if $p < 0.05$. If $p \ge 0.05$, the outcome must be explicitly reported as "no detectable difference".
- **Confidence Intervals**: Wilson 95% score interval for all reported proportions and accuracies.

---

## 6. Extraction Quality Scoring (against `gold_kg.json`)

### Entity Alignment & Metrics
- Extracted canonical entities are mapped to gold entities by exact normalized canonical name or alias match.
- **Entity Recall**: $\frac{\text{Number of gold entities matched by at least one extracted entity}}{\text{Total gold entities}}$.
- **Entity Precision**: $\frac{\text{Number of extracted entities matching a gold entity}}{\text{Total extracted entities}}$ (unmatched extracted entities are counted as spurious).
- **Splits**: Number of gold entities mapped to $\ge 2$ distinct extracted entities.
- **False Merges**: Number of extracted entities mapped to $\ge 2$ distinct gold entities (e.g. distractor name collisions).

### Relationship Alignment & Metrics
- Evaluated over the set of **validated edges** in the knowledge base graph.
- **Exact Relationship Match**: Extracted edge $(u, v, r)$ matches gold edge $(u_{gold}, v_{gold}, r_{gold})$ if mapped source matches $u_{gold}$, mapped target matches $v_{gold}$, and relation type $r == r_{gold}$.
  - Precision and Recall reported.
- **Relaxed Relationship Match**: Extracted edge $(u, v)$ matches gold edge $(u_{gold}, v_{gold})$ if $\{u, v\} == \{u_{gold}, v_{gold}\}$ (ignoring direction and relation type).
  - Precision and Recall reported.

### Gold Edge Fate Breakdown
Every edge in `gold_kg.json` is categorized into exactly one fate:
1. **Validated-Correct**: Extracted and successfully passed validation (matching gold source, target, relation).
2. **Extracted-then-Rejected**: Extracted during stage 1 extraction, but rejected during validation, tabulated with the specific rejection reason code (e.g., `REJECTED_UNKNOWN_TYPE`, `REJECTED_HALLUCINATED_QUOTE`, `REJECTED_ENDPOINTS`, `REJECTED_NEGATION`, `REJECTED_UNCERTAIN`).
3. **Never Extracted**: Never proposed by the extractor LLM during extraction.

---

## 7. Path Coverage Analysis

For each multi-hop question of hop count $h \in \{2, 3\}$:
- The validated extracted knowledge graph is treated as an undirected graph.
- Map the question's first entity $e_{start}$ to the extracted graph, and the gold target entity $e_{target}$ to the extracted graph.
- Determine if a path of length $\le h$ exists between $e_{start}$ and $e_{target}$ in the validated graph.
- Compute:
  1. **Path Coverage**: Fraction of questions in the hop group with an existing path of length $\le h$.
  2. **Conditional Accuracy**: Accuracy of `graph_only` and `hybrid` when path exists vs when path does not exist.
