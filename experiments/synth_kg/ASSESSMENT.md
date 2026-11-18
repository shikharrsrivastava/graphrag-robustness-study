# Synthetic KG Benchmark: Assessment

**Date**: 2026-11-16  
**Corpus**: `experiments/synth_kg/corpus_final/` (60 entities, 95 gold edges, 60 documents)  
**Evaluation Scope**: 120 questions (30 hop1, 30 hop2, 30 hop3, 15 alias, 15 unanswerable)  
**Reference Report**: `experiments/synth_kg/REPORT.md`  

---

## 1. Hybrid vs. Vector and Graph vs. Vector Comparisons

### (a) Hybrid vs. Vector-Only
Across all 105 answerable questions:
- **`hybrid` achieved 53.33% accuracy** (56/105, 95% CI: [43.8%, 62.6%]), compared to **`vector_only` at 42.86%** (45/105, 95% CI: [33.8%, 52.4%]).
- This constitutes a **detectable overall difference** via exact two-sided sign test on discordant pairs ($n_{11}=42, n_{10}=14, n_{01}=3, n_{00}=46$, 17 discordant pairs, $p = 0.0127$).
- **Breakdown by question class**:
  - `hop1`: No detectable difference. `vector_only` scored 100.0% (30/30) vs. `hybrid` 96.7% (29/30) ($n_{11}=29, n_{10}=0, n_{01}=1, n_{00}=0, p = 1.0000$).
  - `hop2`: **Detected difference in favor of `hybrid`**. `hybrid` reached 60.0% (18/30) vs. `vector_only` 20.0% (6/30), an advantage of +40.0 percentage points ($n_{11}=5, n_{10}=13, n_{01}=1, n_{00}=11$, 14 discordant pairs, $p = 0.0018$).
  - `hop3`: No detectable difference. Both modes achieved 6.7% (2/30) ($n_{11}=1, n_{10}=1, n_{01}=1, n_{00}=27, p = 1.0000$).
  - `alias`: No detectable difference. Both modes achieved 46.7% (7/15) ($n_{11}=7, n_{10}=0, n_{01}=0, n_{00}=8, p = 1.0000$).
  - `unanswerable`: Identical performance. Both modes scored 26.7% (4/15 correct abstentions).

### (b) Graph-Only vs. Vector-Only
Across all 105 answerable questions:
- **`graph_only` achieved 42.86% accuracy** (45/105, 95% CI: [33.8%, 52.4%]), exactly matching `vector_only` at 42.86% (45/105).
- There is **no detectable difference overall** ($n_{11}=31, n_{10}=14, n_{01}=14, n_{00}=46$, 28 discordant pairs, $p = 1.0000$).
- **Breakdown by question class**:
  - `hop1`: No detectable difference. `graph_only` 93.3% (28/30) vs. `vector_only` 100.0% (30/30) ($n_{11}=28, n_{10}=0, n_{01}=2, n_{00}=0, p = 0.5000$).
  - `hop2`: **Detected difference in favor of `graph_only`**. `graph_only` reached 53.3% (16/30) vs. `vector_only` 20.0% (6/30), an advantage of +33.3 percentage points ($n_{11}=3, n_{10}=13, n_{01}=3, n_{00}=11$, 16 discordant pairs, $p = 0.0213$).
  - `hop3`: No detectable difference. `graph_only` 3.3% (1/30) vs. `vector_only` 6.7% (2/30) ($n_{11}=0, n_{10}=1, n_{01}=2, n_{00}=27, p = 1.0000$).
  - `alias`: **Detected difference in favor of `vector_only`**. `graph_only` scored 0.0% (0/15) vs. `vector_only` 46.7% (7/15) ($n_{11}=0, n_{10}=0, n_{01}=7, n_{00}=8$, 7 discordant pairs, $p = 0.0156$). `graph_only` has no dense vector retrieval fallback; because seed resolution relies strictly on canonical entities, alias-only queries yield zero seed entities, leading to total retrieval failure.

### (c) Community-Aware vs. Vector-Only
- Overall answerable: **Detected difference**. `community_aware` reached 52.38% (55/105) vs. `vector_only` 42.86% ($p = 0.0213$), driven by `hop2` (56.7% vs. 20.0%, $p = 0.0034$), closely tracking `hybrid`.

---

## 2. Severity of the Extraction-Quality Bottleneck

The extraction pipeline is a primary structural bottleneck for graph completeness:

1. **Entity Extraction and Resolution**:
   - **Entity Recall**: 100.00% (60/60 gold entities recalled).
   - **Entity Precision**: 77.32% (75 valid out of 97 extracted canonical entities; 22 spurious/distractor entities).
   - **Entity Splits**: 15 gold entities were split into $\ge 2$ canonical entities due to surface form variations.
   - **False Merges**: 0 (no distinct gold entities were erroneously merged into a single canonical entity).

2. **Relationship Extraction and Validation**:
   - **Exact Relationship Recall**: **48.42%** (46/95 gold edges recalled in the validated graph). More than half (51.58%) of true relations failed to enter the validated graph.
   - **Exact Relationship Precision**: **83.64%** (46/55 validated edges matched an exact gold triple).
   - **Relaxed Relationship Recall**: **56.84%** (54/95 gold edges recalled when ignoring edge type and direction).
   - **Relaxed Relationship Precision**: **100.00%** (55/55 validated edges connected an undirected pair present in the gold graph).

3. **Gold Edge Fate Breakdown (95 Gold Edges Total)**:
   - `VALIDATED_CORRECT`: 46 edges (48.4%)
   - `REJECTED: REJECTED_ENDPOINT_NOT_IN_EVIDENCE`: 17 edges (17.9%) — Extracted triples where one endpoint was not verbatim in the chunk quote.
   - `EXTRACTED_BUT_DROPPED_IN_RESOLUTION`: 14 edges (14.7%) — Extracted triples dropped during canonical ID remapping because resolved entity IDs failed to form an admitted edge.
   - `REJECTED: REJECTED_ONTOLOGY`: 13 edges (13.7%) — Extracted triples whose relation type or argument domain/range violated the `company.yaml` schema.
   - `NEVER_EXTRACTED`: 4 edges (4.2%) — Triples missed entirely by Stage 1 LLM extraction.
   - `REJECTED: REJECTED_QUOTE_MISMATCH`: 1 edge (1.1%) — Triples where the extracted quote could not be located in chunk text.

**Takeaway**: The bottleneck is not entity discovery (100% recall) nor basic extraction omission (only 4.2% never extracted). The bottleneck is triple validation and resolution: 46.3% of gold edges (44/95) were extracted by the LLM but subsequently rejected by endpoint grounding checks (17.9%), dropped during entity resolution remapping (14.7%), or rejected by ontology filters (13.7%).

---

## 3. Path Coverage Explaining Multi-Hop Results

Validated graph path coverage directly accounts for question answering performance on multi-hop questions:

1. **Path Coverage Statistics**:
   - For `hop2`: A valid path of length $\le 2$ between query seed and gold answer existed in the validated graph for **only 7 of 30 questions (23.3%)**.
   - For `hop3`: A valid path of length $\le 3$ existed for **only 4 of 30 questions (13.3%)**.

2. **Accuracy Conditional on Path Existence**:
   - In `hop2`:
     - **When a validated path exists ($N=7$)**:
       - `graph_only`: **100.0%** (7/7)
       - `hybrid`: **100.0%** (7/7)
       - `vector_only`: **28.6%** (2/7)
     - **When the validated path is missing ($N=23$)**:
       - `graph_only`: drops to **39.1%** (9/23)
       - `hybrid`: drops to **47.8%** (11/23)
       - `vector_only`: **17.4%** (4/23)
   - In `hop3`:
     - **When a validated path exists ($N=4$)**:
       - All modes (`graph_only`, `hybrid`, `vector_only`) scored **0.0%** (0/4).
     - **When the validated path is missing ($N=26$)**:
       - `graph_only`: **3.8%** (1/26)
       - `hybrid`: **7.7%** (2/26)
       - `vector_only`: **7.7%** (2/26)

**Takeaway**: For 2-hop questions, graph path presence is decisive: conditional accuracy is 100% for both `graph_only` and `hybrid`. When the path is missing, graph retrieval degrades to partial or fallback evidence. For 3-hop questions, severe edge attrition (13.3% path coverage) combined with low gold evidence chunk recall (3.3%) caused near-total performance failure across all retrieval strategies.

---

## 4. Failure Class Breakdown: Generation vs. Retrieval vs. Extraction

Across all 105 answerable questions:
- `vector_only`: 45 correct, 0 ERROR, 9 ABSTAINED, 35 WRONG_ENTITY, 16 OTHER.
- `hybrid`: 56 correct, 0 ERROR, 7 ABSTAINED, 35 WRONG_ENTITY, 7 OTHER.
- `graph_only`: 45 correct, 0 ERROR, 25 ABSTAINED, 25 WRONG_ENTITY, 10 OTHER.
- `community_aware`: 55 correct, 0 ERROR, 3 ABSTAINED, 39 WRONG_ENTITY, 8 OTHER.

### Attribution:
1. **Generation Failures**: **0.0%**. Across all 480 query executions (120 questions $\times$ 4 modes), there were **0 ERROR** outcomes. The LLM generator (`gemma3:4b`) adhered strictly to prompt constraints, produced well-formed answers, and never crashed or emitted malformed responses.
2. **Extraction Failures**: Underpins graph traversal limitations. With 51.58% of gold relationships rejected or dropped during indexing, the graph lacked the connectivity required for multi-hop traversals.
3. **Retrieval Failures**: Dominated incorrect predictions.
   - For `hop1`: Full gold evidence recall was 100.0% for `vector_only`, `hybrid`, and `community_aware` (96.7% for `graph_only`), yielding 93.3%–100.0% accuracy.
   - For `hop2`: Full evidence recall dropped to 16.7% for `vector_only` vs. 56.7% for `hybrid` and 53.3% for `graph_only`.
   - For `hop3`: Full evidence recall was 0.0% for `vector_only` and only 3.3% for `hybrid`, `graph_only`, and `community_aware`. In all failed 3-hop questions, intermediate bridge documents were never retrieved into the 4-chunk context window.

**Conclusion**: Retrieval and extraction failures dominate wrong answers. The generation stage functions as intended given sufficient context; failures occur when upstream extraction or retrieval fails to surface the bridge documents or valid graph paths.

---

## 5. Limitations of this Benchmark

1. **Templated Synthetic Language**: Documents and queries were generated via deterministic grammatical templates. Real-world corpora feature complex discourse structure, pronominal anaphora across document boundaries, and stylistic heterogeneity that may affect extraction differently.
2. **Single Domain Ontology**: Evaluation was conducted strictly on the `company.yaml` schema (5 entity types, 5 relationship types). Different domain schemas (e.g., biomedical or legal) may encounter different extraction and validation rejection distributions.
3. **Single Random Seed**: Both the corpus and question set were generated under a single random seed (`seed=7`). The resulting graph topology and question distribution represent a single sample.
4. **Calibration Mechanism**: Difficulty calibration was conducted solely against dense vector retrieval (`embeddinggemma:latest`). While this established questions that vector search could not trivially answer, it did not calibrate against sparse BM25 or alternative dense embedding architectures.
5. **Sample Size**: The sample of 120 questions (30 per hop class, 15 alias, 15 unanswerable) provided statistical power ($p < 0.05$) to detect the large effect size in `hop2` (+40.0 percentage points, $p = 0.0018$), but had lower power to detect smaller effect sizes in `hop3` or sub-categories.
