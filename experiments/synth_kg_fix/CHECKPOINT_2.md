# CHECKPOINT 2 - Phase 2 (dev evaluation on seed 7)

Full outputs: `phase2b_tables.txt` (script `kb_tables.py`), `fix3_dev_test_output.txt` (script `fix3_dev_test.py`).

## 2a Rebuild (path used: stored mentions + stored triples, no re-extraction)
- `rebuild_dev.py` loads `entity_mentions.jsonl` (171) and `triples.jsonl` (98, including 43 rejected with their statuses) from the original KB (read-only), runs the fixed resolution, embeddings, graph build, validation, Leiden and community reports into new KBs.
- Control: with the three new flags off, the rebuild reproduces the original KB exactly (97 entities, 55 relationships, identical mention map).
- `kb_dev` (all three flags on, as in company.yaml): 82 entities, 53 relationships, 36 communities/reports. `kb_dev_merge_only` (merge + initial flags on, revalidation off): 82 entities, 55 relationships.
- LLM use in the rebuild: resolver verification of ambiguous merges (29-30 requests) and community summaries. No extraction, no answer generation.
- Retrieval-only runs use the HEAD code tree for "before", the fixed code for "after", the same cached plans, a stubbed answer generator, and an LLM client that raises on any call (0 errors). Replay of HEAD code reproduces the recorded runs on 105/105 questions.
- Fix 3 cannot be tested on the rebuilt KB (extraction is not re-run); see 2c.

## 2b Before / after (full tables in `phase2b_tables.txt`)
Gold-edge fate (95 gold edges; categories in the header of `kb_tables.py`):
| fate | before | merge+initial only | all fixes |
|---|---|---|---|
| A validated, exact gold type | 27 | 46 | 47 |
| E validated exact, endpoint on a split entity | 19 | 0 | 0 |
| B validated under another type, same endpoints | 17 | 17 | 15 |
| C rejected by ontology | 11 | 11 | 12 |
| D rejected for evidence reasons | 14 | 14 | 14 |
| F never extracted | 7 | 7 | 7 |

Entities: canonical 97 -> 82; gold entities split 15 -> 0; false merges 0 -> 0; extra entities (no gold canonical-name match) 22 -> 22 (unchanged: descriptors, distractor names); gold entities with the wrong canonical type 0 -> 1 (Durnenlok Noxornok, tie broken to PRODUCT).

Alias seeds resolved by the graph route (graph_only records): before 0/15; after 10/15, and in all 10 the resolved seed is exactly the gold entity. The 5 unresolved: 4 "<token> Division" aliases (Q_0093, Q_0100, Q_0101, Q_0102) and the product stem "Haelenrix" (Q_0104). No rule for partial names was added (listed as a limitation).

Retrieval-only all-gold evidence recall at K=4 (Wilson 95%; N = 30/30/30/15):
| mode | run | hop1 | hop2 | hop3 | alias |
|---|---|---|---|---|---|
| hybrid | before | 100.0% | 56.7% [39.2, 72.6] | 3.3% [0.6, 16.7] | 40.0% [19.8, 64.3] |
| hybrid | merge+initial only | 100.0% | 53.3% | 3.3% | 66.7% |
| hybrid | all fixes | 100.0% | 60.0% [42.3, 75.4] | 3.3% [0.6, 16.7] | 66.7% [41.7, 84.8] |
| graph_only | before | 96.7% | 53.3% | 3.3% | 0.0% [0.0, 20.4] |
| graph_only | merge+initial only | 100.0% | 66.7% | 33.3% | 60.0% |
| graph_only | all fixes | 100.0% | 60.0% [42.3, 75.4] | 20.0% [9.5, 37.3] | 60.0% [35.7, 80.2] |
| vector_only | before = after | 100.0% | 16.7% | 0.0% | 40.0% |
Paired changes before -> all fixes (descriptive, unadjusted): hybrid hop2 +1 (Q_0055), alias +4, hop3 0; graph_only alias +9 (sign p = 0.0039), hop3 +5 (p = 0.0625), hop2 +2. No question lost in either mode.

Reading: the alias improvement is large in the graph route and moderate in hybrid (the dense route already found 6 of 15). Hybrid hop3 stays at 1/30 in every variant although graph_only reaches 6-10/30, so in hybrid the graph evidence does not make it into the 4 context chunks for hop3 (merge/RRF ordering; not investigated, not changed). For comparison on seed 7, B9 (graph-free) reaches hop2 21/30, alias 10/15, and B10 reaches hop3 10/30 (B10 was defined after the seed-7 data were seen), so these dev numbers do not show a graph advantage over graph-free baselines.

## 2c Fix 3 dev test (rule fixed in the prompt before the run)
- Documents: T1 two-relation docs (10: DOC_0001..0010), T2 supporting docs of the 14 score.py EXTRACTED_BUT_DROPPED_IN_RESOLUTION gold edges (10 docs; 5 overlap T1), union 15, plus 6 controls (`random.Random(0)`) = 21 documents. Same model/parameters as the original ingestion (gemma3:4b, T=0); stored entity mentions; relationship extraction only.
- Result over all 21 documents: (a) gold edges recovered v1.0 = 18, v1.1 = 19 (rise 1); (b) validated edges matching no gold edge v1.0 = 9, v1.1 = 7 (rise -2). Only-v1.1: EDGE_0047, 0051, 0072; only-v1.0: EDGE_0063, 0068. Documents where the model emitted two relation types for one pair: v1.0 3/10, v1.1 2/10.
- Rule: enable v1.1 only if (a) rises by >= 3 AND (b) rises by no more than (a). (a) rose by 1 -> **KEEP v1.0; v1.1 is not enabled.** `configs/domains/company.yaml` therefore needs no edit (it never sets the version after my A3 change). This is the last config state before the freeze.
- Observation: the selected docs include 20 T1 gold edges but v1.0 already recovers 10 of them; the multi-relation instruction does not raise recall. The previous agent's claim that the prompt is the cause was IMPRECISE (Phase 0).

## Deviations / notes
- T2 contains 10 documents, not 14 (14 edges, 10 documents); T1 is 10 documents (not 9).
- Revalidation costs graph_only traversal links on seed 7 (see CHECKPOINT_1 addendum). Flags remain as specified.
