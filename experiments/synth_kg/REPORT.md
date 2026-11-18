# Synthetic KG Benchmark: Evaluation Report

**Date Generated**: 2026-11-16  
**Status**: **COMPLETED**  
## 1. Experimental Settings Snapshot

| Parameter | Setting / Value |
| :--- | :--- |
| **Public API Facade** | `hybrid_graphrag.HybridGraphRAG` |
| **Base Config** | `configs/default.yaml` |
| **Domain Config** | `configs/domains/company.yaml` |
| **Ontology Domain** | `company_knowledge` (5 entity types, 5 relationship types) |
| **KB Storage Directory** | `experiments/synth_kg/kb/synth_company_kb` |
| **LLM Model** | `gemma3:4b` (Temperature: 0.0) |
| **Embedding Model** | `embeddinggemma:latest` (Dim: 768) |
| **Retrieval Top-K** | `vector_top_k = 4` |
| **Final Context Chunks** | `final_context_chunks = 4` |
| **Comparative Reasoning Mode** | `comparative_reasoning_mode = off` |
| **Graph Context Mode** | `graph_context_mode = off` (False) |
| **Answer Prompt Version** | `v1.1` |
| **Corpus Path** | `experiments/synth_kg/corpus_final/` (60 docs, 95 gold edges) |
| **Questions Count** | 120 total (30 hop1, 30 hop2, 30 hop3, 15 alias, 15 unanswerable) |

## 2. Knowledge Base Build Statistics

| Metric | Count / Value |
| :--- | :--- |
| Documents Ingested | 60 |
| Chunks Created | 60 |
| Chunks vs Documents Ratio | 1.00 chunks/doc |
| Extracted Entity Mentions | 171 |
| Resolved Canonical Entities | 97 |
| Total Extracted Triples | 98 |
| Validated Relationships | 55 |
| Rejected Relationships | 43 |
| Communities Detected | 47 |
| Community Reports Generated | 47 |
| Graph Vertices | 97 |
| Graph Edges | 55 |
| Connected Components | 44 |
| Isolated Nodes | 35 |
| Graph Degree (Min / Mean / Max) | 0 / 1.13 / 5 |

### Build Wall-Clock Timings by Stage

| Pipeline Stage | Duration (seconds) | Duration (minutes) |
| :--- | :--- | :--- |
| `document_loading` | 0.01s | 0.00m |
| `chunking` | 0.00s | 0.00m |
| `entity_extraction` | 572.36s | 9.54m |
| `relationship_extraction` | 885.47s | 14.76m |
| `entity_resolution` | 208.08s | 3.47m |
| `embedding_and_vector_indexing` | 3.50s | 0.06m |
| `graph_construction_and_validation` | 0.03s | 0.00m |
| `community_detection` | 0.01s | 0.00m |
| `community_summarization` | 434.77s | 7.25m |
| `total_indexing` | 2104.26s | 35.07m |
| **Total Build Wall Clock** | **2104.44s** | **35.07m** |

### Rejected Triples by Reason Code

| Rejection Reason | Count | Fraction of Rejected |
| :--- | :--- | :--- |
| `REJECTED_ENDPOINT_NOT_IN_EVIDENCE` | 25 | 58.1% |
| `REJECTED_ONTOLOGY` | 15 | 34.9% |
| `REJECTED_QUOTE_MISMATCH` | 3 | 7.0% |

## 3. Extraction Quality Scoring (against gold_kg.json)

### Entity Resolution Scoring

| Metric | Count / Fraction | Wilson / Notes |
| :--- | :--- | :--- |
| Gold Entities Total | 60 | Ground truth entities |
| Extracted Entities Total | 97 | Extracted canonical entities |
| Recalled Gold Entities | 60 | Matched by $\ge 1$ canonical entity |
| Valid Extracted Entities | 75 | Matched to gold entity |
| **Entity Recall** | **100.00%** | Recalled / Gold Total |
| **Entity Precision** | **77.32%** | Valid / Extracted Total |
| Entity Splits | 15 | 1 gold entity $\rightarrow \ge 2$ extracted |
| Entity False Merges | 0 | 1 extracted entity $\rightarrow \ge 2$ gold |

### Validated Relationship Scoring

| Metric | Count / Score | Interpretation |
| :--- | :--- | :--- |
| Gold Edges Total | 95 | Ground truth edges |
| Validated Edges Total | 55 | Edges in validated graph |
| Exact Matched Validated Edges | 46 | Matches (source, target, relation) |
| Exact Recalled Gold Edges | 46 | Gold edges recalled exactly |
| **Exact Relationship Precision** | **83.64%** | Exact matches / Validated edges |
| **Exact Relationship Recall** | **48.42%** | Recalled exact / Gold edges |
| Relaxed Matched Validated Edges | 55 | Matches undirected {u, v} |
| Relaxed Recalled Gold Edges | 54 | Gold edges recalled relaxed |
| **Relaxed Relationship Precision** | **100.00%** | Relaxed matches / Validated edges |
| **Relaxed Relationship Recall** | **56.84%** | Recalled relaxed / Gold edges |

## 4. Gold Edge Fate Breakdown

| Fate Category | Count | Percentage of Gold Edges |
| :--- | :--- | :--- |
| `VALIDATED_CORRECT` | 46 | 48.4% |
| `REJECTED: REJECTED_ENDPOINT_NOT_IN_EVIDENCE` | 17 | 17.9% |
| `EXTRACTED_BUT_DROPPED_IN_RESOLUTION` | 14 | 14.7% |
| `REJECTED: REJECTED_ONTOLOGY` | 13 | 13.7% |
| `NEVER_EXTRACTED` | 4 | 4.2% |
| `REJECTED: REJECTED_QUOTE_MISMATCH` | 1 | 1.1% |

## 5. Question Answering Performance by Mode and Question Group

### Mode: `vector_only`

- **Overall Answerable Accuracy**: 42.86% (95% CI: [33.8%, 52.4%])
- **Overall Corpus Accuracy (inc. Unanswerable)**: 40.83% (95% CI: [32.5%, 49.8%])

| Question Group | Total | Correct | Accuracy | Wilson 95% CI | ERROR Count | Evidence Recall | Mean Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `hop1` | 30 | 30 | 100.0% | [88.6%, 100.0%] | 0 | 100.0% | 28.04s |
| `hop2` | 30 | 6 | 20.0% | [9.5%, 37.3%] | 0 | 16.7% | 31.27s |
| `hop3` | 30 | 2 | 6.7% | [1.8%, 21.3%] | 0 | 0.0% | 33.83s |
| `alias` | 15 | 7 | 46.7% | [24.8%, 69.9%] | 0 | 40.0% | 31.04s |
| `unanswerable` | 15 | 4 | 26.7% | [10.9%, 52.0%] | 0 | 0.0% | 32.34s |

### Mode: `hybrid`

- **Overall Answerable Accuracy**: 53.33% (95% CI: [43.8%, 62.6%])
- **Overall Corpus Accuracy (inc. Unanswerable)**: 50.00% (95% CI: [41.2%, 58.8%])

| Question Group | Total | Correct | Accuracy | Wilson 95% CI | ERROR Count | Evidence Recall | Mean Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `hop1` | 30 | 29 | 96.7% | [83.3%, 99.4%] | 0 | 100.0% | 38.00s |
| `hop2` | 30 | 18 | 60.0% | [42.3%, 75.4%] | 0 | 56.7% | 43.81s |
| `hop3` | 30 | 2 | 6.7% | [1.8%, 21.3%] | 0 | 3.3% | 47.92s |
| `alias` | 15 | 7 | 46.7% | [24.8%, 69.9%] | 0 | 40.0% | 44.40s |
| `unanswerable` | 15 | 4 | 26.7% | [10.9%, 52.0%] | 0 | 0.0% | 44.66s |

### Mode: `graph_only`

- **Overall Answerable Accuracy**: 42.86% (95% CI: [33.8%, 52.4%])
- **Overall Corpus Accuracy (inc. Unanswerable)**: 40.00% (95% CI: [31.7%, 48.9%])

| Question Group | Total | Correct | Accuracy | Wilson 95% CI | ERROR Count | Evidence Recall | Mean Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `hop1` | 30 | 28 | 93.3% | [78.7%, 98.2%] | 0 | 96.7% | 42.24s |
| `hop2` | 30 | 16 | 53.3% | [36.1%, 69.8%] | 0 | 53.3% | 46.90s |
| `hop3` | 30 | 1 | 3.3% | [0.6%, 16.7%] | 0 | 3.3% | 50.46s |
| `alias` | 15 | 0 | 0.0% | [0.0%, 20.4%] | 0 | 0.0% | 11.57s |
| `unanswerable` | 15 | 3 | 20.0% | [7.0%, 45.2%] | 0 | 0.0% | 42.89s |

### Mode: `community_aware`

- **Overall Answerable Accuracy**: 52.38% (95% CI: [42.9%, 61.7%])
- **Overall Corpus Accuracy (inc. Unanswerable)**: 49.17% (95% CI: [40.4%, 58.0%])

| Question Group | Total | Correct | Accuracy | Wilson 95% CI | ERROR Count | Evidence Recall | Mean Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `hop1` | 30 | 29 | 96.7% | [83.3%, 99.4%] | 0 | 100.0% | 41.83s |
| `hop2` | 30 | 17 | 56.7% | [39.2%, 72.6%] | 0 | 56.7% | 47.73s |
| `hop3` | 30 | 2 | 6.7% | [1.8%, 21.3%] | 0 | 3.3% | 49.60s |
| `alias` | 15 | 7 | 46.7% | [24.8%, 69.9%] | 0 | 40.0% | 45.85s |
| `unanswerable` | 15 | 4 | 26.7% | [10.9%, 52.0%] | 0 | 0.0% | 41.90s |

## 6. Paired Statistical Comparisons vs. `vector_only`

### Candidate: `hybrid` vs. Reference: `vector_only`

| Question Slice | Both Correct ($n_{11}$) | Only Cand ($n_{10}$) | Only Vec ($n_{01}$) | Neither ($n_{00}$) | Discordant | Exact Sign Test $p$-value | Significant ($p < 0.05$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `all_answerable` | 42 | 14 | 3 | 46 | 17 | 0.0127 | **DETECTED** |
| `hop1` | 29 | 0 | 1 | 0 | 1 | 1.0000 | No detectable difference |
| `hop2` | 5 | 13 | 1 | 11 | 14 | 0.0018 | **DETECTED** |
| `hop3` | 1 | 1 | 1 | 27 | 2 | 1.0000 | No detectable difference |
| `alias` | 7 | 0 | 0 | 8 | 0 | 1.0000 | No detectable difference |

### Candidate: `graph_only` vs. Reference: `vector_only`

| Question Slice | Both Correct ($n_{11}$) | Only Cand ($n_{10}$) | Only Vec ($n_{01}$) | Neither ($n_{00}$) | Discordant | Exact Sign Test $p$-value | Significant ($p < 0.05$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `all_answerable` | 31 | 14 | 14 | 46 | 28 | 1.0000 | No detectable difference |
| `hop1` | 28 | 0 | 2 | 0 | 2 | 0.5000 | No detectable difference |
| `hop2` | 3 | 13 | 3 | 11 | 16 | 0.0213 | **DETECTED** |
| `hop3` | 0 | 1 | 2 | 27 | 3 | 1.0000 | No detectable difference |
| `alias` | 0 | 0 | 7 | 8 | 7 | 0.0156 | **DETECTED** |

### Candidate: `community_aware` vs. Reference: `vector_only`

| Question Slice | Both Correct ($n_{11}$) | Only Cand ($n_{10}$) | Only Vec ($n_{01}$) | Neither ($n_{00}$) | Discordant | Exact Sign Test $p$-value | Significant ($p < 0.05$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `all_answerable` | 42 | 13 | 3 | 47 | 16 | 0.0213 | **DETECTED** |
| `hop1` | 29 | 0 | 1 | 0 | 1 | 1.0000 | No detectable difference |
| `hop2` | 5 | 12 | 1 | 12 | 13 | 0.0034 | **DETECTED** |
| `hop3` | 1 | 1 | 1 | 27 | 2 | 1.0000 | No detectable difference |
| `alias` | 7 | 0 | 0 | 8 | 0 | 1.0000 | No detectable difference |

## 7. Gold Evidence Supporting Document Recall

Fraction of questions whose retrieved context chunks contain **all** gold supporting documents:

| Mode | hop1 (1 doc) | hop2 (2 docs) | hop3 (3 docs) | alias (1-2 docs) |
| :--- | :--- | :--- | :--- | :--- |
| `vector_only` | 100.0% | 16.7% | 0.0% | 40.0% |
| `hybrid` | 100.0% | 56.7% | 3.3% | 40.0% |
| `graph_only` | 96.7% | 53.3% | 3.3% | 0.0% |
| `community_aware` | 100.0% | 56.7% | 3.3% | 40.0% |

## 8. Failure Class Breakdown (Answerable Questions)

| Mode | Total Answerable | Correct | ERROR | ABSTAINED | WRONG_ENTITY | OTHER |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `vector_only` | 105 | 45 | 0 | 9 | 35 | 16 |
| `hybrid` | 105 | 56 | 0 | 7 | 35 | 7 |
| `graph_only` | 105 | 45 | 0 | 25 | 25 | 10 |
| `community_aware` | 105 | 55 | 0 | 3 | 39 | 8 |

## 9. Path Coverage and Conditional Accuracy

### Graph Path Coverage on Validated Graph

| Multi-Hop Group | Total Questions | Validated Path Exists (length $\le h$) | Path Coverage Fraction |
| :--- | :--- | :--- | :--- |
| `hop2` | 30 | 7 | 23.3% |
| `hop3` | 30 | 4 | 13.3% |

### Accuracy Conditional on Path Existence in Validated Graph

| Mode & Group | Path Exists (Correct / Total) | Path Exists Accuracy | Path Missing (Correct / Total) | Path Missing Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| `graph_only` (hop2) | 7/7 | 100.0% | 9/23 | 39.1% |
| `graph_only` (hop3) | 0/4 | 0.0% | 1/26 | 3.8% |
| `hybrid` (hop2) | 7/7 | 100.0% | 11/23 | 47.8% |
| `hybrid` (hop3) | 0/4 | 0.0% | 2/26 | 7.7% |
| `vector_only` (hop2) | 2/7 | 28.6% | 4/23 | 17.4% |
| `vector_only` (hop3) | 0/4 | 0.0% | 2/26 | 7.7% |

## 10. Per-Question Results Matrix

| Question ID | Group | vector_only | hybrid | graph_only | community_aware |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Q_0001` | `hop1` | ✓ | ✓ | ✓ | ✓ |
| `Q_0002` | `hop1` | ✓ | ✓ | ✓ | ✓ |
| `Q_0003` | `hop1` | ✓ | ✓ | ✓ | ✓ |
| `Q_0004` | `hop1` | ✓ | ✓ | ✓ | ✓ |
| `Q_0005` | `hop1` | ✓ | ✓ | ✓ | ✓ |
| `Q_0006` | `hop1` | ✓ | ✓ | ✓ | ✓ |
| `Q_0007` | `hop1` | ✓ | ✓ | ✓ | ✓ |
| `Q_0008` | `hop1` | ✓ | ✓ | ✗ | ✓ |
| `Q_0009` | `hop1` | ✓ | ✓ | ✓ | ✓ |
| `Q_0010` | `hop1` | ✓ | ✓ | ✓ | ✓ |
| `Q_0011` | `hop1` | ✓ | ✓ | ✓ | ✓ |
| `Q_0012` | `hop1` | ✓ | ✓ | ✓ | ✓ |
| `Q_0013` | `hop1` | ✓ | ✓ | ✓ | ✓ |
| `Q_0014` | `hop1` | ✓ | ✓ | ✓ | ✓ |
| `Q_0015` | `hop1` | ✓ | ✓ | ✗ | ✓ |
| `Q_0016` | `hop1` | ✓ | ✓ | ✓ | ✓ |
| `Q_0017` | `hop1` | ✓ | ✗ | ✓ | ✓ |
| `Q_0018` | `hop1` | ✓ | ✓ | ✓ | ✓ |
| `Q_0019` | `hop1` | ✓ | ✓ | ✓ | ✓ |
| `Q_0020` | `hop1` | ✓ | ✓ | ✓ | ✓ |
| `Q_0021` | `hop1` | ✓ | ✓ | ✓ | ✓ |
| `Q_0022` | `hop1` | ✓ | ✓ | ✓ | ✓ |
| `Q_0023` | `hop1` | ✓ | ✓ | ✓ | ✓ |
| `Q_0024` | `hop1` | ✓ | ✓ | ✓ | ✓ |
| `Q_0025` | `hop1` | ✓ | ✓ | ✓ | ✓ |
| `Q_0026` | `hop1` | ✓ | ✓ | ✓ | ✓ |
| `Q_0027` | `hop1` | ✓ | ✓ | ✓ | ✓ |
| `Q_0028` | `hop1` | ✓ | ✓ | ✓ | ✓ |
| `Q_0029` | `hop1` | ✓ | ✓ | ✓ | ✗ |
| `Q_0030` | `hop1` | ✓ | ✓ | ✓ | ✓ |
| `Q_0031` | `hop2` | ✗ | ✗ | ✗ | ✗ |
| `Q_0032` | `hop2` | ABS | ✓ | ✓ | ✓ |
| `Q_0033` | `hop2` | ✗ | ✗ | ✓ | ✗ |
| `Q_0034` | `hop2` | ✓ | ✗ | ✗ | ✗ |
| `Q_0035` | `hop2` | ✗ | ✗ | ✗ | ✗ |
| `Q_0036` | `hop2` | ✗ | ✓ | ✗ | ✓ |
| `Q_0037` | `hop2` | ✗ | ✗ | ✗ | ✗ |
| `Q_0038` | `hop2` | ✓ | ✓ | ✓ | ✓ |
| `Q_0039` | `hop2` | ABS | ✓ | ✓ | ✓ |
| `Q_0040` | `hop2` | ✓ | ✓ | ABS | ✓ |
| `Q_0041` | `hop2` | ✗ | ✓ | ✓ | ✓ |
| `Q_0042` | `hop2` | ✗ | ✓ | ✓ | ✓ |
| `Q_0043` | `hop2` | ✗ | ✗ | ABS | ✗ |
| `Q_0044` | `hop2` | ✓ | ✓ | ✓ | ✓ |
| `Q_0045` | `hop2` | ✗ | ✓ | ✓ | ✓ |
| `Q_0046` | `hop2` | ✗ | ✓ | ✓ | ✓ |
| `Q_0047` | `hop2` | ✗ | ABS | ✗ | ✗ |
| `Q_0048` | `hop2` | ✗ | ✗ | ✗ | ✗ |
| `Q_0049` | `hop2` | ✓ | ✓ | ABS | ✓ |
| `Q_0050` | `hop2` | ABS | ✓ | ✓ | ✓ |
| `Q_0051` | `hop2` | ✗ | ✗ | ✗ | ✗ |
| `Q_0052` | `hop2` | ✗ | ✓ | ✓ | ✓ |
| `Q_0053` | `hop2` | ✓ | ✓ | ✓ | ✓ |
| `Q_0054` | `hop2` | ✗ | ✓ | ✓ | ✓ |
| `Q_0055` | `hop2` | ✗ | ✗ | ABS | ✗ |
| `Q_0056` | `hop2` | ✗ | ✓ | ✓ | ✓ |
| `Q_0057` | `hop2` | ✗ | ✓ | ✓ | ✗ |
| `Q_0058` | `hop2` | ABS | ✓ | ✓ | ✓ |
| `Q_0059` | `hop2` | ✗ | ✗ | ✗ | ✗ |
| `Q_0060` | `hop2` | ABS | ✗ | ✗ | ✗ |
| `Q_0061` | `hop3` | ✗ | ✗ | ✗ | ✗ |
| `Q_0062` | `hop3` | ✗ | ✗ | ✗ | ✗ |
| `Q_0063` | `hop3` | ✗ | ✗ | ✗ | ✗ |
| `Q_0064` | `hop3` | ✗ | ✗ | ✗ | ✗ |
| `Q_0065` | `hop3` | ✗ | ABS | ✗ | ✗ |
| `Q_0066` | `hop3` | ✗ | ✗ | ✗ | ✗ |
| `Q_0067` | `hop3` | ✗ | ✗ | ✗ | ✗ |
| `Q_0068` | `hop3` | ✗ | ✓ | ✓ | ✓ |
| `Q_0069` | `hop3` | ✗ | ✗ | ✗ | ✗ |
| `Q_0070` | `hop3` | ABS | ✗ | ✗ | ABS |
| `Q_0071` | `hop3` | ✗ | ✗ | ✗ | ✗ |
| `Q_0072` | `hop3` | ✗ | ABS | ABS | ABS |
| `Q_0073` | `hop3` | ✗ | ✗ | ✗ | ✗ |
| `Q_0074` | `hop3` | ✓ | ✗ | ✗ | ✗ |
| `Q_0075` | `hop3` | ✗ | ✗ | ✗ | ✗ |
| `Q_0076` | `hop3` | ✗ | ✗ | ✗ | ✗ |
| `Q_0077` | `hop3` | ABS | ✗ | ✗ | ✗ |
| `Q_0078` | `hop3` | ✗ | ✗ | ✗ | ✗ |
| `Q_0079` | `hop3` | ✓ | ✓ | ABS | ✓ |
| `Q_0080` | `hop3` | ✗ | ✗ | ✗ | ✗ |
| `Q_0081` | `hop3` | ✗ | ✗ | ABS | ✗ |
| `Q_0082` | `hop3` | ✗ | ✗ | ABS | ABS |
| `Q_0083` | `hop3` | ✗ | ✗ | ✗ | ✗ |
| `Q_0084` | `hop3` | ✗ | ✗ | ABS | ✗ |
| `Q_0085` | `hop3` | ✗ | ABS | ABS | ✗ |
| `Q_0086` | `hop3` | ✗ | ✗ | ✗ | ✗ |
| `Q_0087` | `hop3` | ✗ | ✗ | ✗ | ✗ |
| `Q_0088` | `hop3` | ✗ | ✗ | ✗ | ✗ |
| `Q_0089` | `hop3` | ✗ | ✗ | ✗ | ✗ |
| `Q_0090` | `hop3` | ✗ | ABS | ✗ | ✗ |
| `Q_0091` | `alias` | ✓ | ✓ | ABS | ✓ |
| `Q_0092` | `alias` | ✓ | ✓ | ABS | ✓ |
| `Q_0093` | `alias` | ✗ | ✗ | ABS | ✗ |
| `Q_0094` | `alias` | ✗ | ✗ | ABS | ✗ |
| `Q_0095` | `alias` | ABS | ✗ | ABS | ✗ |
| `Q_0096` | `alias` | ✓ | ✓ | ABS | ✓ |
| `Q_0097` | `alias` | ✓ | ✓ | ABS | ✓ |
| `Q_0098` | `alias` | ✗ | ✗ | ABS | ✗ |
| `Q_0099` | `alias` | ✗ | ✗ | ABS | ✗ |
| `Q_0100` | `alias` | ✓ | ✓ | ABS | ✓ |
| `Q_0101` | `alias` | ✓ | ✓ | ABS | ✓ |
| `Q_0102` | `alias` | ABS | ABS | ABS | ✗ |
| `Q_0103` | `alias` | ✗ | ✗ | ABS | ✗ |
| `Q_0104` | `alias` | ✓ | ✓ | ABS | ✓ |
| `Q_0105` | `alias` | ✗ | ABS | ABS | ✗ |
| `Q_0106` | `unanswerable` | ✗ | ✗ | ✗ | ✓ |
| `Q_0107` | `unanswerable` | ✗ | ✗ | ✗ | ✗ |
| `Q_0108` | `unanswerable` | ✗ | ✗ | ✗ | ✗ |
| `Q_0109` | `unanswerable` | ✗ | ✗ | ✗ | ✗ |
| `Q_0110` | `unanswerable` | ✓ | ✓ | ✓ | ✓ |
| `Q_0111` | `unanswerable` | ✗ | ✗ | ✗ | ✗ |
| `Q_0112` | `unanswerable` | ✗ | ✗ | ✗ | ✗ |
| `Q_0113` | `unanswerable` | ✗ | ✗ | ✗ | ✗ |
| `Q_0114` | `unanswerable` | ✗ | ✗ | ✗ | ✗ |
| `Q_0115` | `unanswerable` | ✗ | ✗ | ✗ | ✗ |
| `Q_0116` | `unanswerable` | ✓ | ✗ | ✗ | ✓ |
| `Q_0117` | `unanswerable` | ✓ | ✗ | ✗ | ✗ |
| `Q_0118` | `unanswerable` | ✗ | ✓ | ✗ | ✗ |
| `Q_0119` | `unanswerable` | ✓ | ✓ | ✓ | ✗ |
| `Q_0120` | `unanswerable` | ✗ | ✓ | ✓ | ✓ |

## 11. Latency by Mode

| Mode | Mean Latency per Question (s) | Total Query Time (s) | Total Query Time (m) |
| :--- | :--- | :--- | :--- |
| `vector_only` | 31.21s | 3745.10s | 62.42m |
| `hybrid` | 43.56s | 5227.77s | 87.13m |
| `graph_only` | 41.71s | 5005.04s | 83.42m |
| `community_aware` | 45.76s | 5491.05s | 91.52m |

## 12. Example Wrong Answers per Mode (Up to 6 per mode)

### Mode: `vector_only` Examples

**[1] Question `Q_0031` (hop2)**: Which client is served by the product developed by Zanistis Infrastructure Group?  
- **Gold Answer**: Brynandor Networks  
- **Prediction**: Zanistis Infrastructure Group serves Durnonvor Aerospace as stated in [D022_C0000] and Ulxathlok Logistics as stated in [D023_C0000].  
- **Failure Class**: `WRONG_ENTITY`  

**[2] Question `Q_0032` (hop2)**: Which client is served by the department that Taelanfen Wyrusnok belongs to?  
- **Gold Answer**: Brynandor Networks  
- **Prediction**: The evidence does not state which client is served by the department that Taelanfen Wyrusnok belongs to. Chunk D012_C0000 describes Taelanfen Wyrusnok's management of Nebothnok Styxinvath, and Chunk D011_C0000 describes Fennikvath Quenindyn's managem  
- **Failure Class**: `ABSTAINED`  

**[3] Question `Q_0033` (hop2)**: Which client is served by the product developed by Iskorlan Applied Automation Wing?  
- **Gold Answer**: Brynandor Networks  
- **Prediction**: Iskorlan Applied Automation Wing operates in production serving Durnonvor Aerospace [D028_C0000]. Iskardyn Pipeline and Lyronvor Nexus also operate in production serving Durnonvor Aerospace [D038_C0000, D044_C0000].  
- **Failure Class**: `WRONG_ENTITY`  

**[4] Question `Q_0035` (hop2)**: Which department does the employee managed by Fennikvath Quenindyn belong to?  
- **Gold Answer**: Durnalnok Analytics Laboratory  
- **Prediction**: Fennikvath Quenindyn manages Taelanfen Wyrusnok, as stated in chunk D011_C0000 [D011_C0000].  
- **Failure Class**: `OTHER`  

**[5] Question `Q_0036` (hop2)**: Which client is served by the product developed by Eldornax Infrastructure Group?  
- **Gold Answer**: Pryxunvek Syndicate  
- **Prediction**: Eldornax Infrastructure Group operates in production serving Quenentor Consortium [D032_C0000].  
- **Failure Class**: `WRONG_ENTITY`  

**[6] Question `Q_0037` (hop2)**: Which client is served by the product developed by Noxontor Foundational Services Unit?  
- **Gold Answer**: Ulxathlok Logistics  
- **Prediction**: Noxontor Foundational Services Unit delivers dedicated platform support to Kaelunkest Syndicate [D030_C0000].  
- **Failure Class**: `WRONG_ENTITY`  

### Mode: `hybrid` Examples

**[1] Question `Q_0017` (hop1)**: What technology is Lyronvor Nexus built with?  
- **Gold Answer**: AethithnokQL  
- **Prediction**: Lyronvor Nexus is built with WyrendorQL, as stated in chunk D043_C0000: "Jorustex Nexus was constructed upon the WyrendorQL framework."  
- **Failure Class**: `WRONG_ENTITY`  

**[2] Question `Q_0031` (hop2)**: Which client is served by the product developed by Zanistis Infrastructure Group?  
- **Gold Answer**: Brynandor Networks  
- **Prediction**: According to [D022_C0000], Zanistis Infrastructure Group provides specialized commercial solutions for Durnonvor Aerospace, indicating that Durnonvor Aerospace is the client served by Zanistis Infrastructure Group.  
- **Failure Class**: `WRONG_ENTITY`  

**[3] Question `Q_0033` (hop2)**: Which client is served by the product developed by Iskorlan Applied Automation Wing?  
- **Gold Answer**: Brynandor Networks  
- **Prediction**: According to [D028_C0000], Iskorlan Applied Automation Wing operates in production serving Durnonvor Aerospace, indicating that Durnonvor Aerospace is the client served by this division.  
- **Failure Class**: `WRONG_ENTITY`  

**[4] Question `Q_0034` (hop2)**: Starting from Rynunvor Lyrontis, traversing MANAGES then MANAGES, what is the target entity?  
- **Gold Answer**: Tarniklan Raelonrak  
- **Prediction**: Starting from Rynunvor Lyrontis, the first MANAGES relationship leads to Zanistis Infrastructure Group. The second MANAGES relationship, originating from Zanistis Infrastructure Group, leads to Durnonvor Aerospace. [GRAPH_EDGE_PROVENANCE: Rynunvor Ly  
- **Failure Class**: `WRONG_ENTITY`  

**[5] Question `Q_0035` (hop2)**: Which department does the employee managed by Fennikvath Quenindyn belong to?  
- **Gold Answer**: Durnalnok Analytics Laboratory  
- **Prediction**: Fennikvath Quenindyn operates as a key component under Eldornax Infrastructure Group and actively manages and directs Taelanfen Wyrusnok [D011_C0000]. Therefore, Taelanfen Wyrusnok belongs to the Eldornax Infrastructure Group [D011_C0000].  
- **Failure Class**: `WRONG_ENTITY`  

**[6] Question `Q_0037` (hop2)**: Which client is served by the product developed by Noxontor Foundational Services Unit?  
- **Gold Answer**: Ulxathlok Logistics  
- **Prediction**: Noxontor Foundational Services Unit delivers dedicated platform support to Kaelunkest Syndicate, according to [D030_C0000].  
- **Failure Class**: `WRONG_ENTITY`  

### Mode: `graph_only` Examples

**[1] Question `Q_0008` (hop1)**: Which client does Zantonnax Infrastructure Group serve?  
- **Gold Answer**: Ulxathlok Logistics  
- **Prediction**: According to [D002_C0000], Gryxethvath Nebikvane is an official organizational part of Zantonnax Infrastructure Group and actively manages and directs it. Therefore, Zantonnax Infrastructure Group serves as the client for Gryxethvath Nebikvane.  
- **Failure Class**: `OTHER`  

**[2] Question `Q_0015` (hop1)**: What product is developed by Oryxonvath Systems Engineering Unit?  
- **Gold Answer**: Sarkinfen Matrix  
- **Prediction**: Oryxonvath Systems Engineering Unit develops Haelenrix Gateway. This is stated in chunk D040_C0000: 'Haelenrix Gateway is engineered primarily using NoxarnokNet. Haelenrix Gateway operates in production serving Kaelunkest Syndicate.'  
- **Failure Class**: `WRONG_ENTITY`  

**[3] Question `Q_0031` (hop2)**: Which client is served by the product developed by Zanistis Infrastructure Group?  
- **Gold Answer**: Brynandor Networks  
- **Prediction**: Zanistis Infrastructure Group provides specialized commercial solutions for Durnonvor Aerospace and directly serves the enterprise requirements of Durnonvor Aerospace, as stated in [D022_C0000]. Therefore, Durnonvor Aerospace is the client served by   
- **Failure Class**: `WRONG_ENTITY`  

**[4] Question `Q_0034` (hop2)**: Starting from Rynunvor Lyrontis, traversing MANAGES then MANAGES, what is the target entity?  
- **Gold Answer**: Tarniklan Raelonrak  
- **Prediction**: Starting from Rynunvor Lyrontis, the first MANAGES relationship leads to Zanistis Infrastructure Group. The second MANAGES relationship, originating from Zanistis Infrastructure Group, results in Durnonvor Aerospace. This is explicitly stated in the   
- **Failure Class**: `WRONG_ENTITY`  

**[5] Question `Q_0035` (hop2)**: Which department does the employee managed by Fennikvath Quenindyn belong to?  
- **Gold Answer**: Durnalnok Analytics Laboratory  
- **Prediction**: According to Evidence Chunk D011_C0000, Fennikvath Quenindyn operates as a key component under Eldornax Infrastructure Group and actively manages and directs Taelanfen Wyrusnok. Therefore, the department managed by Fennikvath Quenindyn is the Eldorna  
- **Failure Class**: `WRONG_ENTITY`  

**[6] Question `Q_0036` (hop2)**: Which client is served by the product developed by Eldornax Infrastructure Group?  
- **Gold Answer**: Pryxunvek Syndicate  
- **Prediction**: Eldornax Infrastructure Group serves Quenentor Consortium as stated in [D032_C0000].  
- **Failure Class**: `WRONG_ENTITY`  

### Mode: `community_aware` Examples

**[1] Question `Q_0029` (hop1)**: Which client does Sarkinfen Matrix serve?  
- **Gold Answer**: Ulxathlok Logistics  
- **Prediction**: Sarkinfen Matrix serves Kaelunkest Syndicate. This is stated in [D039_C0000] which says, "Sarkinfen Matrix is engineered primarily using VandethnokStore. It directly serves the enterprise requirements of Kaelunkest Syndicate." Additionally, [D036_C00  
- **Failure Class**: `WRONG_ENTITY`  

**[2] Question `Q_0031` (hop2)**: Which client is served by the product developed by Zanistis Infrastructure Group?  
- **Gold Answer**: Brynandor Networks  
- **Prediction**: According to [D022_C0000], Zanistis Infrastructure Group provides specialized commercial solutions for Durnonvor Aerospace, indicating that Durnonvor Aerospace is the client served by the product developed by Zanistis Infrastructure Group.  
- **Failure Class**: `WRONG_ENTITY`  

**[3] Question `Q_0033` (hop2)**: Which client is served by the product developed by Iskorlan Applied Automation Wing?  
- **Gold Answer**: Brynandor Networks  
- **Prediction**: The product developed by Iskorlan Applied Automation Wing operates in production serving Durnonvor Aerospace. [D028_C0000]  
- **Failure Class**: `WRONG_ENTITY`  

**[4] Question `Q_0034` (hop2)**: Starting from Rynunvor Lyrontis, traversing MANAGES then MANAGES, what is the target entity?  
- **Gold Answer**: Tarniklan Raelonrak  
- **Prediction**: Starting from Rynunvor Lyrontis, traversing MANAGES then MANAGES leads to Caelonwyn Praxungon. Rynunvor Lyrontis manages Zanistis Infrastructure Group [D016_C0000], and Rynunvor Lyrontis manages Caelonwyn Praxungon [D016_C0000].  
- **Failure Class**: `WRONG_ENTITY`  

**[5] Question `Q_0035` (hop2)**: Which department does the employee managed by Fennikvath Quenindyn belong to?  
- **Gold Answer**: Durnalnok Analytics Laboratory  
- **Prediction**: According to chunk D011_C0000, Fennikvath Quenindyn operates as a key component under Eldornax Infrastructure Group. Therefore, the department managed by Fennikvath Quenindyn is Eldornax Infrastructure Group [D011_C0000].  
- **Failure Class**: `WRONG_ENTITY`  

**[6] Question `Q_0037` (hop2)**: Which client is served by the product developed by Noxontor Foundational Services Unit?  
- **Gold Answer**: Ulxathlok Logistics  
- **Prediction**: The evidence states that Noxontor Foundational Services Unit delivers dedicated platform support to Kaelunkest Syndicate [D030_C0000]. Therefore, Kaelunkest Syndicate is the client served by the product developed by Noxontor Foundational Services Uni  
- **Failure Class**: `WRONG_ENTITY`  
