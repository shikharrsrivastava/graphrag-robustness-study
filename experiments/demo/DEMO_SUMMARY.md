# Hybrid GraphRAG: demo summary

## Architecture (8 lines)
1. Ingest: load -> chunk -> entity extraction -> relationship extraction (gemma3:4b via Ollama, ontology-validated) -> entity resolution.
2. Graph: validated triples become a python-igraph graph; Leiden communities with generated reports.
3. Vectors: chunk embeddings (embeddinggemma, 768d) in ChromaDB.
4. Query: LLM planner (intent, seed entities, target type, relationship path) -> seed resolution against canonical entities.
5. Graph route: traverse from seeds, return chunks that carry supporting edges. Vector route: dense top-k. Community route: report-level context.
6. Merge by chunk ID -> reciprocal rank fusion (k=60) -> context (4 chunks here) -> cited answer or abstention.
7. Modes: `hybrid`, `vector_only`, `graph_only`, `community_aware` through `HybridGraphRAG.query(mode=...)`.
8. Domain-agnostic: entity and relation types come from an ontology config; new behaviour sits behind config flags.

## Results (tables pasted by `build_summary.py` from the report files, not retyped)
All-gold evidence recall at K=4 (every gold supporting document is among the 4 retrieved context documents). Wilson 95% CIs.

**Seed 7 (development corpus, all fixes enabled)** - source: `synth_kg_fix/FINAL_REPORT.md`, section 2b

| mode | run | hop1 | hop2 | hop3 | alias |
|---|---|---|---|---|---|
| hybrid | before | 100.0% | 56.7% [39.2, 72.6] | 3.3% [0.6, 16.7] | 40.0% [19.8, 64.3] |
| hybrid | merge+initial only | 100.0% | 53.3% | 3.3% | 66.7% |
| hybrid | all fixes | 100.0% | 60.0% [42.3, 75.4] | 3.3% [0.6, 16.7] | 66.7% [41.7, 84.8] |
| graph_only | before | 96.7% | 53.3% | 3.3% | 0.0% [0.0, 20.4] |
| graph_only | merge+initial only | 100.0% | 66.7% | 33.3% | 60.0% |
| graph_only | all fixes | 100.0% | 60.0% [42.3, 75.4] | 20.0% [9.5, 37.3] | 60.0% [35.7, 80.2] |
| vector_only | before = after | 100.0% | 16.7% | 0.0% | 40.0% |

**Seed 7, hybrid vs its graph route vs dense (3 of 7 rows)** - source: `synth_kg_fix/diagnosis/DIAGNOSIS.md`, Table 1.1

| list | hop1 | hop2 | hop3 | alias |
|---|---|---|---|---|
| hybrid context (K=4) | 30/30 = 100.0% [88.6, 100.0] | 18/30 = 60.0% [42.3, 75.4] | 1/30 = 3.3% [0.6, 16.7] | 10/15 = 66.7% [41.7, 84.8] |
| graph_only as evaluated (variable length) | 30/30 = 100.0% [88.6, 100.0] | 18/30 = 60.0% [42.3, 75.4] | 6/30 = 20.0% [9.5, 37.3] | 9/15 = 60.0% [35.7, 80.2] |
| vector_only (K=4) | 30/30 = 100.0% [88.6, 100.0] | 5/30 = 16.7% [7.3, 33.6] | 0/30 = 0.0% [0.0, 11.4] | 6/15 = 40.0% [19.8, 64.3] |

**Seed 11 (fresh corpus, pre-registered)** - source: `synth_kg_fix/FINAL_REPORT.md`, section 5

| method | hop1 | hop2 | hop3 | alias |
|---|---|---|---|---|
| hybrid | 100.0% | 36.7% [21.9, 54.5] | 0.0% [0.0, 11.4] | 80.0% [54.8, 93.0] |
| vector_only | 100.0% | 13.3% [5.3, 29.7] | 0.0% | 46.7% [24.8, 69.9] |
| B1 dense top-4 | 100.0% | 13.3% | 0.0% | 46.7% |
| B7 iterative dense | 100.0% | 56.7% [39.2, 72.6] | 3.3% | 60.0% |
| B9 overlap top-1 | 100.0% | 76.7% [59.1, 88.2] | 0.0% | 73.3% [48.0, 89.1] |
| B10 two-step | 100.0% | 50.0% | 16.7% [7.3, 33.6] | 60.0% |

**Seed 11 pre-registered tests (paired exact sign test, Holm over 3)**

| outcome | hybrid | baseline | diff | hybrid only / baseline only | p | p Holm | decision statement |
|---|---|---|---|---|---|---|---|
| P1 hop2 vs B9 | 11/30 | 23/30 | -40.0 pts | 0 / 12 | 0.0005 | 0.0015 | graph-free baseline matches or exceeds hybrid |
| P2 hop3 vs B10 | 0/30 | 5/30 | -16.7 pts | 0 / 5 | 0.0625 | 0.1250 | graph-free baseline matches or exceeds hybrid |
| P3 alias vs B9 | 12/15 | 11/15 | +6.7 pts | 2 / 1 | 1.0000 | 1.0000 | inconclusive |

## What works
- The engine runs end to end through the public API on a laptop (no GPU) and the answers carry chunk-level citations.
- On the 7 demo questions (rule-selected, not hand-picked), strict-correct answers: vector_only 2/7, hybrid 4/7. Gold documents retrieved (excluding the unanswerable question): vector_only 2/6, hybrid 4/6. Wall time for the 14 live queries: 147 s vector_only, 211 s hybrid (includes planning and generation on gemma3:4b).
- On 2-hop questions the graph route lets hybrid retrieve the second document that dense retrieval misses (see Q_0032 and Q_0039 in `demo.py` output, with the graph path that connected them).
- Fixes for entity resolution (person-like types from the ontology, identical-name merge across types) removed all gold-entity splits on seed 7 and raised alias resolution in the graph route; see the seed-7 table above.

## What the evaluation found (including the negative result vs graph-free baselines)
- Hybrid beats vector_only on hop2 on both corpora, but the gain is smaller than that of the graph-free baseline B9 (dense top-1 seed plus capitalized-name overlap expansion). On the fresh seed 11, B9 is better than hybrid on hop2 with Holm-adjusted p as shown in the pre-registered table; the hop3 comparison against the two-step baseline B10 is in the same direction but not significant after correction. **Graph-specific benefit is not demonstrated on this benchmark.**
- Hop3 recall is low in every mode and lowest in hybrid. The diagnosis traced it to the graph route not producing the gold documents (extraction losses) and, in hybrid, to RRF crowding: vector-only chunks outrank graph chunks at ranks 3-5 when the vector list has only 4 entries.
- A root cause of lost edges was found: for DEVELOPS, the extractor reverses the direction, and the ontology correctly rejects the reversed pair. This is a model-following problem, not an ontology defect.
- Demo questions where both answers are wrong are kept on purpose: Q_0031, Q_0061, Q_0106. Q_0106 is unanswerable (the corpus says "serves", not "manages"); both modes answered instead of abstaining.
- Caveats: the answer prompt v1.1 is the one used in the KB config; strict "contains" scoring understates correctness; the demo is 7 questions and demonstrates behaviour, it is not a benchmark.

## Next step: direction repair confirmed on fresh seed 13
Candidate fix 1 from the diagnosis: when a triple is rejected by the ontology, swap source and target if the swapped pair satisfies the same relation's rule. The stored-triple computation on seed 7 (exploratory, seed already used) gave:

> triples with final status REJECTED_ONTOLOGY: 17; swap passes the rule: 10; of these, swapped triple equals a gold edge (same relation): 10; swapped triple matches no gold edge: 0
> exact validated gold edges (A+E) before: 47; after swap (upper bound, ignores the D-fate edges and endpoint splits): 56

Seed 11 was consumed by the diagnosis, so confirmation needs a new seed. Plan: (1) write the protocol and decision rules first, behind a config flag with the default unchanged; (2) generate seed 13 with the same knobs as seeds 7 and 11 and freeze the diff hash; (3) ingest once, evaluate retrieval-only with equal limits, Wilson CIs, paired exact sign tests and Holm correction; (4) report against B9 and B10 whatever the result. No tuning after seeing seed 13.
