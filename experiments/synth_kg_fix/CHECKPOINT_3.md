# CHECKPOINT 3 - Phase 3 (fresh-seed confirmation, seed 11)

Protocol: `PROTOCOL.md` (written before any generation). Full outputs: `seed11_eval_output.txt` (pre-registered tests), `seed11_tables.txt` (descriptive tables), `seed11_per_question.json`. Scripts: `ingest_seed11.py`, `embed_corpus.py`, `plan_cache.py`, `retrieval_eval.py`, `seed11_eval.py`, `kb_tables.py`.

## What was run
- Corpus: `generate.py --seed 11 --n_docs 60 --alias_rate 0.3 --pronoun_rate 0.2 --distractor_rate 0.3` -> 60 docs, 60 gold entities, 95 gold edges, 120 questions (30/30/30/15 + 15 unanswerable); `validate.py`: PASSED (first call failed on an import path because `PYTHONPATH` was not set; rerun with it, no code change).
- Ingestion: single attempt through the public API, 2501.7 s, 0 extraction failures. 166 mentions, 97 raw triples (45 VALIDATED, 24 REJECTED_ONTOLOGY, 23 REJECTED_ENDPOINT_NOT_IN_EVIDENCE, 5 REJECTED_QUOTE_MISMATCH), 79 canonical entities, 45 graph relationships, 37 communities. Flags on: merge across types, revalidation, initial+surname merge; prompt v1.0; person-like = EMPLOYEE.
- Evaluation: retrieval only (cached planner output, answer generator stubbed, LLM client raises on any call; 0 errors, 0 planner failures).
- Integrity: `git diff` hash before and after are identical (`9f79ab6d...39ab4`), supplementary all-files hash identical (`73e95fa2...76fe54`). No protocol violation. Test suite after the run: see `final_test_counts.txt`.

## Result: all-gold evidence recall at K=4 (Wilson 95%)
| method | hop1 | hop2 | hop3 | alias |
|---|---|---|---|---|
| hybrid | 100.0% | 36.7% [21.9, 54.5] | 0.0% [0.0, 11.4] | 80.0% [54.8, 93.0] |
| vector_only | 100.0% | 13.3% [5.3, 29.7] | 0.0% | 46.7% [24.8, 69.9] |
| B1 dense top-4 | 100.0% | 13.3% | 0.0% | 46.7% |
| B7 iterative dense | 100.0% | 56.7% [39.2, 72.6] | 3.3% | 60.0% |
| B9 overlap top-1 | 100.0% | 76.7% [59.1, 88.2] | 0.0% | 73.3% [48.0, 89.1] |
| B10 two-step | 100.0% | 50.0% | 16.7% [7.3, 33.6] | 60.0% |
(graph_only, descriptive: hop2 50.0%, hop3 3.3%, alias 66.7%.)

## Pre-registered primary outcomes (paired exact sign test, Holm over 3)
| outcome | hybrid | baseline | diff | hybrid only / baseline only | p | p Holm | decision statement |
|---|---|---|---|---|---|---|---|
| P1 hop2 vs B9 | 11/30 | 23/30 | -40.0 pts | 0 / 12 | 0.0005 | 0.0015 | graph-free baseline matches or exceeds hybrid |
| P2 hop3 vs B10 | 0/30 | 5/30 | -16.7 pts | 0 / 5 | 0.0625 | 0.1250 | graph-free baseline matches or exceeds hybrid |
| P3 alias vs B9 | 12/15 | 11/15 | +6.7 pts | 2 / 1 | 1.0000 | 1.0000 | inconclusive |

No outcome meets "graph advantage demonstrated". On P1 the baseline is better than hybrid with Holm-adjusted p = 0.0015.

## Descriptive (no decision)
- hybrid vs vector_only: hop2 hybrid-only 7, vector-only 0 (unadjusted sign p = 0.0156); alias 5 / 0 (p = 0.0625); hop1 and hop3 no difference. The hybrid gain over dense retrieval is present but smaller than the gain of the graph-free B9.
- Gold-edge fate (95 edges): exact type validated 42, other-type 9, rejected by ontology 22, rejected for evidence 14, never extracted 8; 0 gold entities split; 0 false merges; 19 extra entities; 3 gold TECHNOLOGY entities end with canonical type PRODUCT (NeborzorVM, UlxantisStore, KaeluthvekDB).
- Alias seeds (graph route): 13/15 resolved, in all 13 the resolved seed is exactly the gold entity. The 2 unresolved are a product stem ("Synaldor") and a "<token> Division" form (Q_0091, Q_0094), the same partial-name classes that remained unresolved on seed 7.
- Revalidation on seed 11 changed 4 triples: 2 rejected -> validated, 2 validated -> rejected (one of them BUILT_WITH between PRODUCT and PRODUCT, a consequence of the TECHNOLOGY mistyping above). 16 entities had mixed mention types: 5 resolved by plurality, 4 by relationship validity, 7 by ontology declaration order.
- 11 of the 15 gold DEVELOPS edges are rejected by the ontology (fate C); the cause was not investigated in this task.

## Skipped / failed
- Nothing skipped. No seed other than 11 was generated; seed 11 was ingested and evaluated once.
