# Hybrid GraphRAG: project report

Status as of 2026-11-19, branch `hybrid-graphrag-build`. All tables in section 4, and every measured number in sections 5 and 6, are copied by `experiments/demo/build_report.py` from `experiments/synth_kg_fix/FINAL_REPORT.md` and `experiments/synth_kg_fix/diagnosis/DIAGNOSIS.md`. The prose in this file contains no hand-typed results.

## 1. Goal and scope
The project builds a modular, domain-agnostic Hybrid GraphRAG engine that runs locally (gemma3:4b and embeddinggemma through Ollama, a MacBook Air M2 with 16 GB RAM, no GPU) and evaluates it with scientific discipline. The question the evaluation asks is specific: does adding a knowledge graph to dense retrieval improve multi-hop retrieval, and does it do so beyond what simple graph-free baselines achieve?

In scope: ingestion, entity and relationship extraction, entity resolution, graph construction and validation, communities, hybrid retrieval, cited answers or abstention, and a retrieval-only evaluation harness. Out of scope: large-scale corpora, cloud models, and answer-quality evaluation of the final fixes (only retrieval was evaluated after the fixes).

## 2. Architecture
**Ingestion pipeline.** load -> chunk -> entity extraction -> relationship extraction (validated against an ontology) -> entity resolution -> graph build and validation (python-igraph) -> chunk embeddings (ChromaDB) -> Leiden communities with generated reports.

**Query pipeline.** An LLM planner produces intent, seed entities, a target type and a relationship path. Seeds are resolved against canonical entities. Three routes produce candidate chunks: graph (traversal from the seeds), vector (dense top-k) and community (reports). Candidates are merged by chunk ID, fused with reciprocal rank fusion (k = 60), cut to the context size, and passed to the answer generator, which returns a cited answer or abstains.

**Query modes** (`HybridGraphRAG.query(question, mode=...)`): `hybrid`, `vector_only`, `graph_only`, `community_aware`.

**Public API** (`hybrid_graphrag.HybridGraphRAG`): `from_config`, `from_existing`, `ingest`, `query`, `get_kb_stats`, `close`.

**Modules** (`src/`): `ingestion`, `extraction` (entity and relation extractors, schemas), `resolution` (entity resolver), `graph` (builder, validator, traversal, clustering), `embeddings`, `communities`, `retrieval` (planner, seed retriever, context builder), `generation` (answer generator), `storage`, `core` (configuration, models, ontology, interfaces), and `engine.py` (orchestration). Entity and relation types come from an ontology config (`configs/domains/`), and new behaviour is added behind config flags with unchanged defaults.

## 3. Evaluation method
- **Benchmark.** HotpotQA was tried first and abandoned for this setup: ingestion was too slow and most extracted relationships were rejected. The project moved to a procedural synthetic benchmark (`experiments/synth_kg/`): a company ontology, 60 documents, a gold knowledge graph, and 120 questions in groups hop1, hop2, hop3, alias and unanswerable.
- **Seeds.** Seed 7 is the development corpus. Seed 11 is a fresh corpus used once as a pre-registered confirmation; it was later consumed by the diagnosis (section 6), so it cannot confirm anything further.
- **Metric.** Retrieval-only, document-level all-gold recall at K = 4 (every gold supporting document is among the 4 context documents). Query plans are cached, answer generation is stubbed, and the LLM client raises on any call during evaluation.
- **Hygiene rules.** ERROR is counted separately from abstentions; temperature 0; equal limits across modes; Wilson 95% intervals; exact sign tests on paired discordant questions; Holm correction across families of tests; decision rules written before the run; no tuning after seeing results; scripts that produced reported numbers are not edited afterwards.
- **Baselines.** Graph-free baselines were defined to test whether the graph itself matters: B1 dense top-4, B7 iterative dense, B9 dense top-1 seed with capitalized-name overlap expansion, B10 two-step.
- **Integrity.** The `git diff` hash was frozen before the seed-11 run and checked after (hashes below).

```
Before generation (`frozen_diff_sha.txt`):
```
frozen at 2026-11-16T19:59:13+0530 (before seed-11 corpus generation)
HEAD=3877a4d02be92ca499fbb001e3a7afad971d4a4a
git_diff_sha256=9f79ab6d30ead1b9332071374ba4794d5ed03f4d28ee82ca75a08e0499439ab4
all_files_sha256=73e95fa2af0c6797537454f5bfbf9522ef3b7a2cacbd8364f4ecb4008276fe54
note: git_diff_sha256 is the hash specified in the task (git diff -- src tests configs prompts | shasum -a 256); it does not cover untracked files (tests/test_synth_kg_fixes.py, prompts/relationship_extraction_v1_1.txt). all_files_sha256 is a supplementary hash over every file under src tests configs prompts.
```

After the final evaluation (`frozen_diff_sha_after.txt`):
```
git_diff_sha256=9f79ab6d30ead1b9332071374ba4794d5ed03f4d28ee82ca75a08e0499439ab4
all_files_sha256=73e95fa2af0c6797537454f5bfbf9522ef3b7a2cacbd8364f4ecb4008276fe54
```

Both hashes are identical: no protocol violation.

```

## 4. Results (pasted by script)
### 4.1 Seed 7 (development; retrieval-only, K = 4; before vs after the fixes)
| mode | run | hop1 | hop2 | hop3 | alias |
|---|---|---|---|---|---|
| hybrid | before | 100.0% | 56.7% [39.2, 72.6] | 3.3% [0.6, 16.7] | 40.0% [19.8, 64.3] |
| hybrid | merge+initial only | 100.0% | 53.3% | 3.3% | 66.7% |
| hybrid | all fixes | 100.0% | 60.0% [42.3, 75.4] | 3.3% [0.6, 16.7] | 66.7% [41.7, 84.8] |
| graph_only | before | 96.7% | 53.3% | 3.3% | 0.0% [0.0, 20.4] |
| graph_only | merge+initial only | 100.0% | 66.7% | 33.3% | 60.0% |
| graph_only | all fixes | 100.0% | 60.0% [42.3, 75.4] | 20.0% [9.5, 37.3] | 60.0% [35.7, 80.2] |
| vector_only | before = after | 100.0% | 16.7% | 0.0% | 40.0% |

Gold-edge fate on seed 7 (95 gold edges):

| fate | before | merge+initial only | all fixes |
|---|---|---|---|
| A validated, exact gold type | 27 | 46 | 47 |
| E validated exact, endpoint on a split entity | 19 | 0 | 0 |
| B validated under another type, same endpoints | 17 | 17 | 15 |
| C rejected by ontology | 11 | 11 | 12 |
| D rejected for evidence reasons | 14 | 14 | 14 |
| F never extracted | 7 | 7 | 7 |

### 4.2 Seed 7, candidate lists (source: DIAGNOSIS.md, Table 1.1, seed 7 `kb_dev` only)
| list | hop1 | hop2 | hop3 | alias |
|---|---|---|---|---|
| hybrid context (K=4) | 30/30 = 100.0% [88.6, 100.0] | 18/30 = 60.0% [42.3, 75.4] | 1/30 = 3.3% [0.6, 16.7] | 10/15 = 66.7% [41.7, 84.8] |
| graph_only as evaluated (variable length) | 30/30 = 100.0% [88.6, 100.0] | 18/30 = 60.0% [42.3, 75.4] | 6/30 = 20.0% [9.5, 37.3] | 9/15 = 60.0% [35.7, 80.2] |
| graph_only first 4 chunks | 30/30 = 100.0% [88.6, 100.0] | 18/30 = 60.0% [42.3, 75.4] | 4/30 = 13.3% [5.3, 29.7] | 9/15 = 60.0% [35.7, 80.2] |
| graph list in hybrid, first 4 | 30/30 = 100.0% [88.6, 100.0] | 18/30 = 60.0% [42.3, 75.4] | 4/30 = 13.3% [5.3, 29.7] | 9/15 = 60.0% [35.7, 80.2] |
| graph list in hybrid, full | 30/30 = 100.0% [88.6, 100.0] | 18/30 = 60.0% [42.3, 75.4] | 6/30 = 20.0% [9.5, 37.3] | 9/15 = 60.0% [35.7, 80.2] |
| vector_only (K=4) | 30/30 = 100.0% [88.6, 100.0] | 5/30 = 16.7% [7.3, 33.6] | 0/30 = 0.0% [0.0, 11.4] | 6/15 = 40.0% [19.8, 64.3] |
| union(graph full, vector 4) | 30/30 = 100.0% [88.6, 100.0] | 20/30 = 66.7% [48.8, 80.8] | 6/30 = 20.0% [9.5, 37.3] | 12/15 = 80.0% [54.8, 93.0] |

### 4.3 Seed 11 (fresh corpus, pre-registered)
- Corpus: `generate.py --seed 11 --n_docs 60 --alias_rate 0.3 --pronoun_rate 0.2 --distractor_rate 0.3` -> 60 docs, 60 gold entities, 95 gold edges, 120 questions (30/30/30/15 + 15 unanswerable); `validate.py`: PASSED (first call failed on an import path because `PYTHONPATH` was not set; rerun with it, no code change).

- Ingestion: single attempt through the public API, 2501.7 s, 0 extraction failures. 166 mentions, 97 raw triples (45 VALIDATED, 24 REJECTED_ONTOLOGY, 23 REJECTED_ENDPOINT_NOT_IN_EVIDENCE, 5 REJECTED_QUOTE_MISMATCH), 79 canonical entities, 45 graph relationships, 37 communities. Flags on: merge across types, revalidation, initial+surname merge; prompt v1.0; person-like = EMPLOYEE.

| method | hop1 | hop2 | hop3 | alias |
|---|---|---|---|---|
| hybrid | 100.0% | 36.7% [21.9, 54.5] | 0.0% [0.0, 11.4] | 80.0% [54.8, 93.0] |
| vector_only | 100.0% | 13.3% [5.3, 29.7] | 0.0% | 46.7% [24.8, 69.9] |
| B1 dense top-4 | 100.0% | 13.3% | 0.0% | 46.7% |
| B7 iterative dense | 100.0% | 56.7% [39.2, 72.6] | 3.3% | 60.0% |
| B9 overlap top-1 | 100.0% | 76.7% [59.1, 88.2] | 0.0% | 73.3% [48.0, 89.1] |
| B10 two-step | 100.0% | 50.0% | 16.7% [7.3, 33.6] | 60.0% |

Pre-registered primary outcomes (paired exact sign test, Holm over 3):

| outcome | hybrid | baseline | diff | hybrid only / baseline only | p | p Holm | decision statement |
|---|---|---|---|---|---|---|---|
| P1 hop2 vs B9 | 11/30 | 23/30 | -40.0 pts | 0 / 12 | 0.0005 | 0.0015 | graph-free baseline matches or exceeds hybrid |
| P2 hop3 vs B10 | 0/30 | 5/30 | -16.7 pts | 0 / 5 | 0.0625 | 0.1250 | graph-free baseline matches or exceeds hybrid |
| P3 alias vs B9 | 12/15 | 11/15 | +6.7 pts | 2 / 1 | 1.0000 | 1.0000 | inconclusive |

Descriptive findings reported with those results:

> - hybrid vs vector_only: hop2 hybrid-only 7, vector-only 0 (unadjusted sign p = 0.0156); alias 5 / 0 (p = 0.0625); hop1 and hop3 no difference. The hybrid gain over dense retrieval is present but smaller than the gain of the graph-free B9.
> - Gold-edge fate (95 edges): exact type validated 42, other-type 9, rejected by ontology 22, rejected for evidence 14, never extracted 8; 0 gold entities split; 0 false merges; 19 extra entities; 3 gold TECHNOLOGY entities end with canonical type PRODUCT (NeborzorVM, UlxantisStore, KaeluthvekDB).
> - Alias seeds (graph route): 13/15 resolved, in all 13 the resolved seed is exactly the gold entity. The 2 unresolved are a product stem ("Synaldor") and a "<token> Division" form (Q_0091, Q_0094), the same partial-name classes that remained unresolved on seed 7.
> - Revalidation on seed 11 changed 4 triples: 2 rejected -> validated, 2 validated -> rejected (one of them BUILT_WITH between PRODUCT and PRODUCT, a consequence of the TECHNOLOGY mistyping above). 16 entities had mixed mention types: 5 resolved by plurality, 4 by relationship validity, 7 by ontology declaration order.
> - 11 of the 15 gold DEVELOPS edges are rejected by the ontology (fate C); the cause was not investigated in this task.

## 5. Defects found and fixed, with measured effect
Four defects were verified in the engine and fixed behind config flags (defaults unchanged). The fixes were evaluated as bundles, so per-fix attribution is limited to what the runs below separate. The 'merge+initial only' run is identical to 'all fixes' except that post-resolution revalidation is off.

| defect / fix | what changed | measured effect (source lines below) |
|---|---|---|
| Hardcoded `"PERSON"` literals in the resolver and seed retriever | `person_like_types` is read from the ontology (company ontology: EMPLOYEE); literals replaced in `entity_resolver.py`, `seed_retriever.py`, `relation_extractor.py` | enables surname and initial matching for non-PERSON ontologies; not separable from the merge fix in the runs |
| Initial+surname matching | `person_names_compatible` accepts initial+surname; merge of such mentions behind `enable_initial_surname_person_merge` | no measurable effect: no document in these corpora contains an initial+surname form (FINAL_REPORT section 6, item 2) |
| Entities split by inconsistent mention types | `merge_exact_name_across_types` with a type vote and ontology-order tie-break | see the entity and fate lines below (gold entities split, edge fates, alias seeds) |
| Relationship ontology validation ran on mention types before resolution | `revalidate_relationships_after_resolution` recomputes validity with canonical types | trade-off, see 'Revalidation' below |
| Provenance marker text leaks into answers | `graph_context_mode: "off"` omits the graph path block | not evaluated: no answer generation was run after this change |

Measured on seed 7 (copied from FINAL_REPORT.md section 2b):

- Entities: canonical 97 -> 82; gold entities split 15 -> 0; false merges 0 -> 0; extra entities (no gold canonical-name match) 22 -> 22 (unchanged: descriptors, distractor names); gold entities with the wrong canonical type 0 -> 1 (Durnenlok Noxornok, tie broken to PRODUCT).
- Alias seeds resolved by the graph route (graph_only records): before 0/15; after 10/15, and in all 10 the resolved seed is exactly the gold entity. The 5 unresolved: 4 "<token> Division" aliases (Q_0093, Q_0100, Q_0101, Q_0102) and the product stem "Haelenrix" (Q_0104). No rule for partial names was added (listed as a limitation).
- Paired changes before -> all fixes (descriptive, unadjusted): hybrid hop2 +1 (Q_0055), alias +4, hop3 0; graph_only alias +9 (sign p = 0.0039), hop3 +5 (p = 0.0625), hop2 +2. No question lost in either mode.
- Reading: the alias improvement is large in the graph route and moderate in hybrid (the dense route already found 6 of 15). Hybrid hop3 stays at 1/30 in every variant although graph_only reaches 6-10/30, so in hybrid the graph evidence does not make it into the 4 context chunks for hop3 (merge/RRF ordering; not investigated, not changed). For comparison on seed 7, B9 (graph-free) reaches hop2 21/30, alias 10/15, and B10 reaches hop3 10/30 (B10 was defined after the seed-7 data were seen), so these dev numbers do not show a graph advantage over graph-free baselines.

Revalidation: compare the 'merge+initial only' and 'all fixes' rows in the table in 4.1. The report states the trade-off explicitly: revalidation removes wrong-type edges and also removes some traversal links, so graph_only hop3 is lower with it. Seed 11 shows the same mechanism on a smaller number of changed triples (see the descriptive findings in 4.3).

Prompt v1.1 (a multi-relation instruction, defect 4 in the original list) failed its pre-written dev rule and stays disabled; it was evaluated on a subset of seed-7 documents only.

## 6. Findings
1. **Negative result against graph-free baselines.** On the fresh seed, the pre-registered decisions were that a graph-free baseline matches or exceeds hybrid on hop2 (P1) and hop3 (P2), and inconclusive on alias (P3); see the table in 4.3. Hybrid does beat plain dense retrieval on hop2, but a name-overlap baseline beats hybrid. **Graph-specific benefit is not demonstrated on this benchmark.** The benchmark's procedurally unique names favour name-overlap baselines, which limits how far this generalizes in either direction.
2. **Hop3 fails mainly because of extraction coverage, which is the ceiling.** Three separate effects, of very different size. **(1) Most hop3 questions fail in every mode because the graph route never produces the gold documents.** For 24/30 questions on seed 7 (`kb_dev`) and 28/30 on seed 11, at least one gold document is in neither the graph-route list nor the vector-route list, so no re-ranking of those lists can recover the question (graph_only fails the same questions). The graph route is limited by the stored graph: the engine's own traverser, run uncapped and undirected within 3 hops, covers all gold documents for only 7/30 (seed 7) and 2/30 (seed 11) hop3 questions, and neither the 20-path cap nor edge direction explains it (cap reached in 0 and 0 hop3 queries; directed and capped coverage 6/30 and 1/30, i.e. direction changes coverage by at most one question). A hop3 gold path has 3 edges; all three are exact validated edges for only 1/30 (seed 7) and 0/30 (seed 11) questions. **(2) The gap between graph_only and hybrid is a K mismatch plus RRF crowding.** `graph_only` returns the whole graph list (up to 16 chunks; more than 4 chunks in 19/30 seed-7 and 19/30 seed-11 hop3 contexts), while hybrid is cut to 4 chunks. With graph_only cut to its first 4 chunks, hop3 is 4/30 instead of 6/30 (seed 7 `kb_dev`), 4/30 instead of 10/30 (merge-only KB) and 1/30 instead of 1/30 (seed 11). Hybrid scores 1, 1 and 0. For the questions graph_only wins and hybrid loses (5 on seed 7, 9 on the merge-only KB, 1 on seed 11), the mechanism is RRF crowding in 4, 4 and 1 cases (all missing gold documents have graph rank <= 4 but lose slots to chunks that carry a vector rank: vector-only chunks, or chunks with a low graph rank and a vector rank) and a K cap in 1, 5 and 0 cases (a gold document sits at graph rank 5 or lower). Crowding works like this: the vector list has only 4 entries (`vector_top_k=4`) and RRF scores a vector-only chunk at rank r the same as a graph-only chunk at rank r, so vector ranks 2-4 outrank graph ranks 3-5; in 10 of 11 (seed 7) and 2 of 2 (seed 11) cases where a gold document was in the graph list but not in the context, a vector-only chunk with a higher RRF score was in the context. **(3) The other candidate mechanisms were checked and are not responsible:** merge-by-chunk-id collisions 0/105 (seed 7) and 0/105 (seed 11) (each document has exactly one chunk), ContextBuilder drops a context chunk in 0 and 0 questions (max context 12000-char budget never approached), merged top-4 equals the returned context in 105/105 and 105/105 questions.
3. **Reversed edge direction in DEVELOPS extraction.** The extractor assigns the right relation type and the right entity types but reverses the direction. On seed 11, 11 of the 15 gold DEVELOPS edges (fate C, rejected by ontology) were extracted as `PRODUCT -DEVELOPS-> DEPARTMENT`, while the sentence in the document is active with the department as subject (for example "X designs and engineers Y"). Over both seeds 23 of 30 extracted DEVELOPS triples joining the endpoints of a gold DEVELOPS edge are reversed, against MANAGES 5/30, BELONGS_TO 1/15, BUILT_WITH 5/21, SERVES 0/45 for the other relations (same-relation triples only). The ontology correctly rejects the reversed pair, so this is the extractor not following the stated direction, not an ontology defect.
4. **Ontology-guided direction repair, computed on stored triples (exploratory; the seed was already used, so it is not evidence of a fix):**

> triples with final status REJECTED_ONTOLOGY: 17; swap passes the rule: 10; of these, swapped triple equals a gold edge (same relation): 10; swapped triple matches no gold edge: 0
> exact validated gold edges (A+E) before: 47; after swap (upper bound, ignores the D-fate edges and endpoint splits): 56

5. **Protocol consequence.** **Seed 11 has now been used for diagnosis, so it can no longer serve as a clean confirmation set.** Any future fix must be confirmed on a new seed (use **13**) generated with the same knobs as seeds 7 and 11 (`--n_docs 60 --alias_rate 0.3 --pronoun_rate 0.2 --distractor_rate 0.3`) and **pre-registered (PROTOCOL.md written, frozen hash recorded) before the fix is applied**. The counterfactuals in this report were computed on seeds 7 and 11 only and are labelled *exploratory, not a fix, seed already used*.

## 7. Limitations
See `docs/KNOWN_LIMITATIONS.md` (general design limits and the synthetic-KG audit findings in its last section) and FINAL_REPORT.md section 7 for the evaluation-specific limits. In short: one ontology; two seeds with wide intervals; procedurally unique names; flags chosen on development data; retrieval metric only; the revalidation trade-off; the PRODUCT/TECHNOLOGY type-vote weakness; no coreference step; partial-name aliases unresolved; a remaining hardcoded `"PROJECT"` literal in `src/resolution/entity_resolver.py`.

## 8. Next steps
1. **Direction repair behind a flag** (default off): swap source and target of an ontology-rejected triple when the swapped pair satisfies the same relation's rule.
2. **Confirm on fresh seed 13**, generated with the same knobs as seeds 7 and 11, with the protocol and decision rules written first, the diff hash frozen, one ingestion, retrieval-only evaluation against B9 and B10 using the same statistics as above. No tuning after seeing seed 13.
3. Open and not yet investigated: why hybrid hop3 is far below graph_only (RRF crowding and K mismatch were measured in the diagnosis, no fix applied); the type-vote tie-break for PRODUCT versus TECHNOLOGY; partial-name aliases; coreference.
4. Housekeeping: reusability across ontologies (open-domain config), documentation and release.

## Reproduce
Demo on the seed-7 fixed knowledge base (commands from `experiments/demo/README.md`):

```bash
# default: offline, reads experiments/demo/precomputed.json
python3 experiments/demo/demo.py

# live: runs the engine for real (Ollama must be running with gemma3:4b and embeddinggemma; about 6 minutes)
python3 experiments/demo/demo.py --live
```
