# Diagnosis: hybrid hop3 recall and rejected DEVELOPS edges

Status: diagnosis only. No file under `src/`, `tests/`, `configs/`, `prompts/` or `docs/` was changed, nothing was committed. Every number below is produced by a script in `experiments/synth_kg_fix/diagnosis/` and pasted by `build_diagnosis.py`.

## 0. Protocol note (read first)

**Seed 11 has now been used for diagnosis, so it can no longer serve as a clean confirmation set.** Any future fix must be confirmed on a new seed (use **13**) generated with the same knobs as seeds 7 and 11 (`--n_docs 60 --alias_rate 0.3 --pronoun_rate 0.2 --distractor_rate 0.3`) and **pre-registered (PROTOCOL.md written, frozen hash recorded) before the fix is applied**. The counterfactuals in this report were computed on seeds 7 and 11 only and are labelled *exploratory, not a fix, seed already used*.

Integrity hashes (`experiments/synth_kg_fix/freeze_hash.sh`):

```
before: git_diff_sha256=9f79ab6d30ead1b9332071374ba4794d5ed03f4d28ee82ca75a08e0499439ab4
        all_files_sha256=73e95fa2af0c6797537454f5bfbf9522ef3b7a2cacbd8364f4ecb4008276fe54
after:  git_diff_sha256=9f79ab6d30ead1b9332071374ba4794d5ed03f4d28ee82ca75a08e0499439ab4
        all_files_sha256=73e95fa2af0c6797537454f5bfbf9522ef3b7a2cacbd8364f4ecb4008276fe54
```
Result: **both hashes unchanged**.

How the data was produced. `part1_record.py` runs the engine's query path unchanged (public API `HybridGraphRAG.from_existing`, cached query plans, stubbed answer generator, LLM client that raises on any call; embeddings are the only model call, for the query vector) and wraps three class methods inside the script process to record the graph-route list, the vector-route list and the full merged RRF ranking. Fidelity check: the recorded 4-chunk contexts equal the earlier recorded runs for 315/315 (seed 7, kb_dev), 210/210 (seed 7, merge-only KB), 315/315 (seed 11) (question, mode) pairs.

Side effect to note: opening a knowledge base through the public API updates the modification time of its Chroma files (`vector_store/chroma.sqlite3` and index `.bin` files in `kb_dev` and `kb_dev_merge_only`, which are this task's own derived copies); the same happened in the earlier retrieval runs. No other file outside `diagnosis/` was written, and `src/`, `tests/`, `configs/`, `prompts/` are unchanged (hashes above).

Knowledge bases: seed 7 = `kb_dev` (all fixes enabled, 53 relationships); seed 7 merge-only = `kb_dev_merge_only` (55 relationships, no revalidation); seed 11 = `kb_seed11`. The "6-10/30" graph_only figure in FINAL_REPORT.md for seed 7 is therefore 6/30 on `kb_dev` and 10/30 on `kb_dev_merge_only`; both are analysed below.

## Part 1. Why is hybrid hop3 evidence recall (0-1/30) far below graph_only?

### Finding

Three separate effects, of very different size. **(1) Most hop3 questions fail in every mode because the graph route never produces the gold documents.** For 24/30 questions on seed 7 (`kb_dev`) and 28/30 on seed 11, at least one gold document is in neither the graph-route list nor the vector-route list, so no re-ranking of those lists can recover the question (graph_only fails the same questions). The graph route is limited by the stored graph: the engine's own traverser, run uncapped and undirected within 3 hops, covers all gold documents for only 7/30 (seed 7) and 2/30 (seed 11) hop3 questions, and neither the 20-path cap nor edge direction explains it (cap reached in 0 and 0 hop3 queries; directed and capped coverage 6/30 and 1/30, i.e. direction changes coverage by at most one question). A hop3 gold path has 3 edges; all three are exact validated edges for only 1/30 (seed 7) and 0/30 (seed 11) questions. **(2) The gap between graph_only and hybrid is a K mismatch plus RRF crowding.** `graph_only` returns the whole graph list (up to 16 chunks; more than 4 chunks in 19/30 seed-7 and 19/30 seed-11 hop3 contexts), while hybrid is cut to 4 chunks. With graph_only cut to its first 4 chunks, hop3 is 4/30 instead of 6/30 (seed 7 `kb_dev`), 4/30 instead of 10/30 (merge-only KB) and 1/30 instead of 1/30 (seed 11). Hybrid scores 1, 1 and 0. For the questions graph_only wins and hybrid loses (5 on seed 7, 9 on the merge-only KB, 1 on seed 11), the mechanism is RRF crowding in 4, 4 and 1 cases (all missing gold documents have graph rank <= 4 but lose slots to chunks that carry a vector rank: vector-only chunks, or chunks with a low graph rank and a vector rank) and a K cap in 1, 5 and 0 cases (a gold document sits at graph rank 5 or lower). Crowding works like this: the vector list has only 4 entries (`vector_top_k=4`) and RRF scores a vector-only chunk at rank r the same as a graph-only chunk at rank r, so vector ranks 2-4 outrank graph ranks 3-5; in 10 of 11 (seed 7) and 2 of 2 (seed 11) cases where a gold document was in the graph list but not in the context, a vector-only chunk with a higher RRF score was in the context. **(3) The other candidate mechanisms were checked and are not responsible:** merge-by-chunk-id collisions 0/105 (seed 7) and 0/105 (seed 11) (each document has exactly one chunk), ContextBuilder drops a context chunk in 0 and 0 questions (max context 12000-char budget never approached), merged top-4 equals the returned context in 105/105 and 105/105 questions.

### Table 1.1 All-gold recall by candidate list (answerable questions), seeds 7 / 7 merge-only / 11

**seed 7 (kb_dev)**

| list | hop1 | hop2 | hop3 | alias |
|---|---|---|---|---|
| hybrid context (K=4) | 30/30 = 100.0% [88.6, 100.0] | 18/30 = 60.0% [42.3, 75.4] | 1/30 = 3.3% [0.6, 16.7] | 10/15 = 66.7% [41.7, 84.8] |
| graph_only as evaluated (variable length) | 30/30 = 100.0% [88.6, 100.0] | 18/30 = 60.0% [42.3, 75.4] | 6/30 = 20.0% [9.5, 37.3] | 9/15 = 60.0% [35.7, 80.2] |
| graph_only first 4 chunks | 30/30 = 100.0% [88.6, 100.0] | 18/30 = 60.0% [42.3, 75.4] | 4/30 = 13.3% [5.3, 29.7] | 9/15 = 60.0% [35.7, 80.2] |
| graph list in hybrid, first 4 | 30/30 = 100.0% [88.6, 100.0] | 18/30 = 60.0% [42.3, 75.4] | 4/30 = 13.3% [5.3, 29.7] | 9/15 = 60.0% [35.7, 80.2] |
| graph list in hybrid, full | 30/30 = 100.0% [88.6, 100.0] | 18/30 = 60.0% [42.3, 75.4] | 6/30 = 20.0% [9.5, 37.3] | 9/15 = 60.0% [35.7, 80.2] |
| vector_only (K=4) | 30/30 = 100.0% [88.6, 100.0] | 5/30 = 16.7% [7.3, 33.6] | 0/30 = 0.0% [0.0, 11.4] | 6/15 = 40.0% [19.8, 64.3] |
| union(graph full, vector 4) | 30/30 = 100.0% [88.6, 100.0] | 20/30 = 66.7% [48.8, 80.8] | 6/30 = 20.0% [9.5, 37.3] | 12/15 = 80.0% [54.8, 93.0] |
graph_only context length (chunks) per group: hop1: min 2 median 7 max 16 share>4: 24/30; hop2: min 2 median 8 max 14 share>4: 21/30; hop3: min 2 median 8 max 16 share>4: 19/30; alias: min 0 median 3 max 12 share>4: 4/15

graph_only context length (chunks) per group: hop1: min 2 median 7 max 16 share>4: 24/30; hop2: min 2 median 8 max 14 share>4: 21/30; hop3: min 2 median 8 max 16 share>4: 19/30; alias: min 0 median 3 max 12 share>4: 4/15

**seed 7 merge-only KB**

| list | hop1 | hop2 | hop3 | alias |
|---|---|---|---|---|
| hybrid context (K=4) | 30/30 = 100.0% [88.6, 100.0] | 16/30 = 53.3% [36.1, 69.8] | 1/30 = 3.3% [0.6, 16.7] | 10/15 = 66.7% [41.7, 84.8] |
| graph_only as evaluated (variable length) | 30/30 = 100.0% [88.6, 100.0] | 20/30 = 66.7% [48.8, 80.8] | 10/30 = 33.3% [19.2, 51.2] | 9/15 = 60.0% [35.7, 80.2] |
| graph_only first 4 chunks | 30/30 = 100.0% [88.6, 100.0] | 18/30 = 60.0% [42.3, 75.4] | 4/30 = 13.3% [5.3, 29.7] | 9/15 = 60.0% [35.7, 80.2] |
| graph list in hybrid, first 4 | 30/30 = 100.0% [88.6, 100.0] | 18/30 = 60.0% [42.3, 75.4] | 4/30 = 13.3% [5.3, 29.7] | 9/15 = 60.0% [35.7, 80.2] |
| graph list in hybrid, full | 30/30 = 100.0% [88.6, 100.0] | 20/30 = 66.7% [48.8, 80.8] | 10/30 = 33.3% [19.2, 51.2] | 9/15 = 60.0% [35.7, 80.2] |
| vector_only (K=4) | 30/30 = 100.0% [88.6, 100.0] | 5/30 = 16.7% [7.3, 33.6] | 0/30 = 0.0% [0.0, 11.4] | 6/15 = 40.0% [19.8, 64.3] |
| union(graph full, vector 4) | 30/30 = 100.0% [88.6, 100.0] | 22/30 = 73.3% [55.6, 85.8] | 10/30 = 33.3% [19.2, 51.2] | 12/15 = 80.0% [54.8, 93.0] |
graph_only context length (chunks) per group: hop1: min 1 median 7 max 16 share>4: 23/30; hop2: min 1 median 8 max 19 share>4: 21/30; hop3: min 1 median 9 max 19 share>4: 19/30; alias: min 0 median 3 max 12 share>4: 4/15

graph_only context length (chunks) per group: hop1: min 1 median 7 max 16 share>4: 23/30; hop2: min 1 median 8 max 19 share>4: 21/30; hop3: min 1 median 9 max 19 share>4: 19/30; alias: min 0 median 3 max 12 share>4: 4/15

**seed 11**

| list | hop1 | hop2 | hop3 | alias |
|---|---|---|---|---|
| hybrid context (K=4) | 30/30 = 100.0% [88.6, 100.0] | 11/30 = 36.7% [21.9, 54.5] | 0/30 = 0.0% [0.0, 11.4] | 12/15 = 80.0% [54.8, 93.0] |
| graph_only as evaluated (variable length) | 30/30 = 100.0% [88.6, 100.0] | 15/30 = 50.0% [33.2, 66.8] | 1/30 = 3.3% [0.6, 16.7] | 10/15 = 66.7% [41.7, 84.8] |
| graph_only first 4 chunks | 30/30 = 100.0% [88.6, 100.0] | 15/30 = 50.0% [33.2, 66.8] | 1/30 = 3.3% [0.6, 16.7] | 10/15 = 66.7% [41.7, 84.8] |
| graph list in hybrid, first 4 | 30/30 = 100.0% [88.6, 100.0] | 15/30 = 50.0% [33.2, 66.8] | 1/30 = 3.3% [0.6, 16.7] | 10/15 = 66.7% [41.7, 84.8] |
| graph list in hybrid, full | 30/30 = 100.0% [88.6, 100.0] | 15/30 = 50.0% [33.2, 66.8] | 1/30 = 3.3% [0.6, 16.7] | 10/15 = 66.7% [41.7, 84.8] |
| vector_only (K=4) | 30/30 = 100.0% [88.6, 100.0] | 4/30 = 13.3% [5.3, 29.7] | 0/30 = 0.0% [0.0, 11.4] | 7/15 = 46.7% [24.8, 69.9] |
| union(graph full, vector 4) | 30/30 = 100.0% [88.6, 100.0] | 15/30 = 50.0% [33.2, 66.8] | 2/30 = 6.7% [1.8, 21.3] | 12/15 = 80.0% [54.8, 93.0] |
graph_only context length (chunks) per group: hop1: min 1 median 7 max 16 share>4: 21/30; hop2: min 1 median 7 max 16 share>4: 21/30; hop3: min 1 median 7 max 10 share>4: 19/30; alias: min 0 median 7 max 16 share>4: 9/15

graph_only context length (chunks) per group: hop1: min 1 median 7 max 16 share>4: 21/30; hop2: min 1 median 7 max 16 share>4: 21/30; hop3: min 1 median 7 max 10 share>4: 19/30; alias: min 0 median 7 max 16 share>4: 9/15

### Table 1.2 Hop3: graph_only vs hybrid and mechanism for every failing question

**seed 7 (kb_dev)**

```
TABLE 2 (seed 7): hop3 questions, graph_only vs hybrid
  N=30; both succeed 1; graph_only only 5; hybrid only 0; neither 24
  mechanism for every hop3 question (hybrid misses >=1 gold doc), seed 7
     24  A: gold doc in neither route list (candidate generation)
      4  B: RRF crowding (gold within graph top-4 but displaced by other chunks)
      1  C: K cap (gold at graph-list rank > 4, cannot fit 4 slots with vector chunks)
      1  hybrid succeeds
  mechanism restricted to the questions where graph_only succeeds and hybrid does not:
      4  B: RRF crowding (gold within graph top-4 but displaced by other chunks)
      1  C: K cap (gold at graph-list rank > 4, cannot fit 4 slots with vector chunks)
```

| question | graph_only len | gold ranks in graph list | gold ranks in vector list | gold merged positions | hybrid context docs | mechanism |
|---|---|---|---|---|---|---|
| Q_0062 | 16 | [2, 3, 4] | [1, None, None] | [1, 5, 6] | D013(GRAPH+VECTOR),D012(GRAPH+VECTOR),D028(GRAPH+VECTOR),D002(VECTOR) | B: RRF crowding (gold within graph top-4 but displaced by other chunks) |
| Q_0064 | 8 | [2, 3, 5] | [1, None, None] | [2, 3, 7] | D017(GRAPH+VECTOR),D018(GRAPH+VECTOR),D019(GRAPH),D014(VECTOR) | C: K cap (gold at graph-list rank > 4, cannot fit 4 slots with vector chunks) |
| Q_0066 | 8 | [2, 3, 4] | [1, None, None] | [2, 3, 5] | D017(GRAPH+VECTOR),D018(GRAPH+VECTOR),D019(GRAPH),D014(VECTOR) | B: RRF crowding (gold within graph top-4 but displaced by other chunks) |
| Q_0067 | 12 | [1, 3, 4] | [1, 2, None] | [1, 2, 5] | D005(GRAPH+VECTOR),D026(GRAPH+VECTOR),D020(GRAPH+VECTOR),D029(VECTOR) | B: RRF crowding (gold within graph top-4 but displaced by other chunks) |
| Q_0089 | 16 | [2, 3, 10] | [1, None, 4] | [1, 5, 2] | D013(GRAPH+VECTOR),D015(GRAPH+VECTOR),D012(GRAPH),D002(VECTOR) | B: RRF crowding (gold within graph top-4 but displaced by other chunks) |

```
  displacing chunks for gold docs that are in the graph list but not in the hybrid context (all hop3):
    Q_0062 gold DOC_0014: graph rank 3, merged position 5, score 0.01587; context = D013[GRAPH+VECTOR g2 v1 0.03252]; D012[GRAPH+VECTOR g1 v3 0.03227]; D028[GRAPH+VECTOR g13 v4 0.02932]; D002[VECTOR gNone v2 0.01613]
    Q_0062 gold DOC_0035: graph rank 4, merged position 6, score 0.01562; context = D013[GRAPH+VECTOR g2 v1 0.03252]; D012[GRAPH+VECTOR g1 v3 0.03227]; D028[GRAPH+VECTOR g13 v4 0.02932]; D002[VECTOR gNone v2 0.01613]
    Q_0064 gold DOC_0025: graph rank 5, merged position 7, score 0.01538; context = D017[GRAPH+VECTOR g1 v2 0.03252]; D018[GRAPH+VECTOR g2 v1 0.03252]; D019[GRAPH g3 vNone 0.01587]; D014[VECTOR gNone v3 0.01587]
    Q_0066 gold DOC_0020: graph rank 4, merged position 5, score 0.01562; context = D017[GRAPH+VECTOR g1 v2 0.03252]; D018[GRAPH+VECTOR g2 v1 0.03252]; D019[GRAPH g3 vNone 0.01587]; D014[VECTOR gNone v3 0.01587]
    Q_0067 gold DOC_0041: graph rank 4, merged position 5, score 0.01562; context = D005[GRAPH+VECTOR g1 v1 0.03279]; D026[GRAPH+VECTOR g3 v2 0.03200]; D020[GRAPH+VECTOR g2 v4 0.03175]; D029[VECTOR gNone v3 0.01587]
    Q_0069 gold DOC_0027: graph rank 4, merged position 6, score 0.01562; context = D021[GRAPH+VECTOR g2 v2 0.03226]; D020[GRAPH g1 vNone 0.01639]; D031[VECTOR gNone v1 0.01639]; D006[GRAPH g3 vNone 0.01587]
    Q_0075 gold DOC_0022: graph rank 4, merged position 6, score 0.01562; context = D016[GRAPH+VECTOR g2 v1 0.03252]; D015[GRAPH+VECTOR g1 v3 0.03227]; D028[GRAPH+VECTOR g5 v4 0.03101]; D044[GRAPH+VECTOR g8 v2 0.03083]
    Q_0083 gold DOC_0015: graph rank 9, merged position 11, score 0.01449; context = D013[GRAPH+VECTOR g1 v2 0.03252]; D014[GRAPH+VECTOR g2 v1 0.03252]; D035[GRAPH g3 vNone 0.01587]; D018[VECTOR gNone v3 0.01587]
    Q_0087 gold DOC_0027: graph rank 4, merged position 6, score 0.01562; context = D021[GRAPH+VECTOR g2 v2 0.03226]; D020[GRAPH g1 vNone 0.01639]; D031[VECTOR gNone v1 0.01639]; D006[GRAPH g3 vNone 0.01587]
    Q_0089 gold DOC_0014: graph rank 3, merged position 5, score 0.01587; context = D013[GRAPH+VECTOR g2 v1 0.03252]; D015[GRAPH+VECTOR g10 v4 0.02991]; D012[GRAPH g1 vNone 0.01639]; D002[VECTOR gNone v2 0.01613]
    Q_0090 gold DOC_0025: graph rank 4, merged position 5, score 0.01562; context = D019[GRAPH+VECTOR g2 v1 0.03252]; D018[GRAPH+VECTOR g1 v3 0.03227]; D028[VECTOR gNone v2 0.01613]; D004[GRAPH g3 vNone 0.01587]
```

**seed 7 merge-only KB**

```
TABLE 2 (seed 7mo): hop3 questions, graph_only vs hybrid
  N=30; both succeed 1; graph_only only 9; hybrid only 0; neither 20
  mechanism for every hop3 question (hybrid misses >=1 gold doc), seed 7mo
     20  A: gold doc in neither route list (candidate generation)
      4  B: RRF crowding (gold within graph top-4 but displaced by other chunks)
      5  C: K cap (gold at graph-list rank > 4, cannot fit 4 slots with vector chunks)
      1  hybrid succeeds
  mechanism restricted to the questions where graph_only succeeds and hybrid does not:
      4  B: RRF crowding (gold within graph top-4 but displaced by other chunks)
      5  C: K cap (gold at graph-list rank > 4, cannot fit 4 slots with vector chunks)
```

| question | graph_only len | gold ranks in graph list | gold ranks in vector list | gold merged positions | hybrid context docs | mechanism |
|---|---|---|---|---|---|---|
| Q_0062 | 16 | [2, 3, 4] | [1, None, None] | [1, 5, 6] | D013(GRAPH+VECTOR),D012(GRAPH+VECTOR),D028(GRAPH+VECTOR),D002(VECTOR) | B: RRF crowding (gold within graph top-4 but displaced by other chunks) |
| Q_0064 | 8 | [2, 3, 5] | [1, None, None] | [2, 3, 7] | D017(GRAPH+VECTOR),D018(GRAPH+VECTOR),D019(GRAPH),D014(VECTOR) | C: K cap (gold at graph-list rank > 4, cannot fit 4 slots with vector chunks) |
| Q_0066 | 8 | [2, 3, 4] | [1, None, None] | [2, 3, 5] | D017(GRAPH+VECTOR),D018(GRAPH+VECTOR),D019(GRAPH),D014(VECTOR) | B: RRF crowding (gold within graph top-4 but displaced by other chunks) |
| Q_0067 | 12 | [1, 3, 4] | [1, 2, None] | [1, 2, 5] | D005(GRAPH+VECTOR),D026(GRAPH+VECTOR),D020(GRAPH+VECTOR),D029(VECTOR) | B: RRF crowding (gold within graph top-4 but displaced by other chunks) |
| Q_0070 | 11 | [1, 3, 7] | [1, None, None] | [1, 4, 10] | D011(GRAPH+VECTOR),D032(GRAPH),D026(VECTOR),D012(GRAPH) | C: K cap (gold at graph-list rank > 4, cannot fit 4 slots with vector chunks) |
| Q_0076 | 17 | [2, 4, 5] | [1, None, None] | [1, 7, 8] | D012(GRAPH+VECTOR),D014(GRAPH+VECTOR),D015(GRAPH+VECTOR),D011(GRAPH) | C: K cap (gold at graph-list rank > 4, cannot fit 4 slots with vector chunks) |
| Q_0080 | 19 | [2, 10, 12] | [1, None, None] | [2, 12, 14] | D015(GRAPH+VECTOR),D016(GRAPH+VECTOR),D001(GRAPH),D020(VECTOR) | C: K cap (gold at graph-list rank > 4, cannot fit 4 slots with vector chunks) |
| Q_0086 | 19 | [2, 5, 7] | [1, None, None] | [2, 7, 9] | D015(GRAPH+VECTOR),D016(GRAPH+VECTOR),D044(GRAPH+VECTOR),D001(GRAPH) | C: K cap (gold at graph-list rank > 4, cannot fit 4 slots with vector chunks) |
| Q_0089 | 16 | [2, 3, 10] | [1, None, 4] | [1, 5, 2] | D013(GRAPH+VECTOR),D015(GRAPH+VECTOR),D012(GRAPH),D002(VECTOR) | B: RRF crowding (gold within graph top-4 but displaced by other chunks) |

```
  displacing chunks for gold docs that are in the graph list but not in the hybrid context (all hop3):
    Q_0062 gold DOC_0014: graph rank 3, merged position 5, score 0.01587; context = D013[GRAPH+VECTOR g2 v1 0.03252]; D012[GRAPH+VECTOR g1 v3 0.03227]; D028[GRAPH+VECTOR g13 v4 0.02932]; D002[VECTOR gNone v2 0.01613]
    Q_0062 gold DOC_0035: graph rank 4, merged position 6, score 0.01562; context = D013[GRAPH+VECTOR g2 v1 0.03252]; D012[GRAPH+VECTOR g1 v3 0.03227]; D028[GRAPH+VECTOR g13 v4 0.02932]; D002[VECTOR gNone v2 0.01613]
    Q_0064 gold DOC_0025: graph rank 5, merged position 7, score 0.01538; context = D017[GRAPH+VECTOR g1 v2 0.03252]; D018[GRAPH+VECTOR g2 v1 0.03252]; D019[GRAPH g3 vNone 0.01587]; D014[VECTOR gNone v3 0.01587]
    Q_0065 gold DOC_0033: graph rank 4, merged position 6, score 0.01562; context = D012[GRAPH+VECTOR g2 v1 0.03252]; D028[GRAPH+VECTOR g13 v2 0.02983]; D011[GRAPH g1 vNone 0.01639]; D013[GRAPH g3 vNone 0.01587]
    Q_0066 gold DOC_0020: graph rank 4, merged position 5, score 0.01562; context = D017[GRAPH+VECTOR g1 v2 0.03252]; D018[GRAPH+VECTOR g2 v1 0.03252]; D019[GRAPH g3 vNone 0.01587]; D014[VECTOR gNone v3 0.01587]
    Q_0067 gold DOC_0041: graph rank 4, merged position 5, score 0.01562; context = D005[GRAPH+VECTOR g1 v1 0.03279]; D026[GRAPH+VECTOR g3 v2 0.03200]; D020[GRAPH+VECTOR g2 v4 0.03175]; D029[VECTOR gNone v3 0.01587]
    Q_0069 gold DOC_0027: graph rank 4, merged position 6, score 0.01562; context = D021[GRAPH+VECTOR g2 v2 0.03226]; D020[GRAPH g1 vNone 0.01639]; D031[VECTOR gNone v1 0.01639]; D006[GRAPH g3 vNone 0.01587]
    Q_0070 gold DOC_0033: graph rank 7, merged position 10, score 0.01493; context = D011[GRAPH+VECTOR g1 v1 0.03279]; D032[GRAPH g2 vNone 0.01613]; D026[VECTOR gNone v2 0.01613]; D012[GRAPH g3 vNone 0.01587]
    Q_0075 gold DOC_0022: graph rank 5, merged position 7, score 0.01538; context = D016[GRAPH+VECTOR g2 v1 0.03252]; D015[GRAPH+VECTOR g1 v3 0.03227]; D028[GRAPH+VECTOR g9 v4 0.03012]; D044[GRAPH+VECTOR g12 v2 0.03002]
    Q_0076 gold DOC_0013: graph rank 4, merged position 7, score 0.01562; context = D012[GRAPH+VECTOR g2 v1 0.03252]; D014[GRAPH+VECTOR g15 v3 0.02921]; D015[GRAPH+VECTOR g16 v4 0.02878]; D011[GRAPH g1 vNone 0.01639]
    Q_0076 gold DOC_0034: graph rank 5, merged position 8, score 0.01538; context = D012[GRAPH+VECTOR g2 v1 0.03252]; D014[GRAPH+VECTOR g15 v3 0.02921]; D015[GRAPH+VECTOR g16 v4 0.02878]; D011[GRAPH g1 vNone 0.01639]
    Q_0080 gold DOC_0017: graph rank 10, merged position 12, score 0.01429; context = D015[GRAPH+VECTOR g1 v2 0.03252]; D016[GRAPH+VECTOR g2 v1 0.03252]; D001[GRAPH g3 vNone 0.01587]; D020[VECTOR gNone v3 0.01587]
    Q_0080 gold DOC_0018: graph rank 12, merged position 14, score 0.01389; context = D015[GRAPH+VECTOR g1 v2 0.03252]; D016[GRAPH+VECTOR g2 v1 0.03252]; D001[GRAPH g3 vNone 0.01587]; D020[VECTOR gNone v3 0.01587]
    Q_0083 gold DOC_0015: graph rank 9, merged position 11, score 0.01449; context = D013[GRAPH+VECTOR g1 v2 0.03252]; D014[GRAPH+VECTOR g2 v1 0.03252]; D035[GRAPH g3 vNone 0.01587]; D018[VECTOR gNone v3 0.01587]
    Q_0086 gold DOC_0017: graph rank 5, merged position 7, score 0.01538; context = D015[GRAPH+VECTOR g1 v2 0.03252]; D016[GRAPH+VECTOR g2 v1 0.03252]; D044[GRAPH+VECTOR g16 v3 0.02903]; D001[GRAPH g3 vNone 0.01587]
    Q_0086 gold DOC_0023: graph rank 7, merged position 9, score 0.01493; context = D015[GRAPH+VECTOR g1 v2 0.03252]; D016[GRAPH+VECTOR g2 v1 0.03252]; D044[GRAPH+VECTOR g16 v3 0.02903]; D001[GRAPH g3 vNone 0.01587]
    Q_0087 gold DOC_0027: graph rank 4, merged position 6, score 0.01562; context = D021[GRAPH+VECTOR g2 v2 0.03226]; D020[GRAPH g1 vNone 0.01639]; D031[VECTOR gNone v1 0.01639]; D006[GRAPH g3 vNone 0.01587]
    Q_0089 gold DOC_0014: graph rank 3, merged position 5, score 0.01587; context = D013[GRAPH+VECTOR g2 v1 0.03252]; D015[GRAPH+VECTOR g10 v4 0.02991]; D012[GRAPH g1 vNone 0.01639]; D002[VECTOR gNone v2 0.01613]
    Q_0090 gold DOC_0025: graph rank 4, merged position 5, score 0.01562; context = D019[GRAPH+VECTOR g2 v1 0.03252]; D018[GRAPH+VECTOR g1 v3 0.03227]; D028[VECTOR gNone v2 0.01613]; D004[GRAPH g3 vNone 0.01587]
```

**seed 11**

```
TABLE 2 (seed 11): hop3 questions, graph_only vs hybrid
  N=30; both succeed 0; graph_only only 1; hybrid only 0; neither 29
  mechanism for every hop3 question (hybrid misses >=1 gold doc), seed 11
     28  A: gold doc in neither route list (candidate generation)
      1  B: RRF crowding (gold within graph top-4 but displaced by other chunks)
      1  D: gold only in vector list or mixed, not selected
  mechanism restricted to the questions where graph_only succeeds and hybrid does not:
      1  B: RRF crowding (gold within graph top-4 but displaced by other chunks)
```

| question | graph_only len | gold ranks in graph list | gold ranks in vector list | gold merged positions | hybrid context docs | mechanism |
|---|---|---|---|---|---|---|
| Q_0073 | 4 | [1, 3, 4] | [1, 2, None] | [1, 2, 5] | D005(GRAPH+VECTOR),D026(GRAPH+VECTOR),D020(GRAPH),D036(VECTOR) | B: RRF crowding (gold within graph top-4 but displaced by other chunks) |

```
  displacing chunks for gold docs that are in the graph list but not in the hybrid context (all hop3):
    Q_0073 gold DOC_0041: graph rank 4, merged position 5, score 0.01562; context = D005[GRAPH+VECTOR g1 v1 0.03279]; D026[GRAPH+VECTOR g3 v2 0.03200]; D020[GRAPH g2 vNone 0.01613]; D036[VECTOR gNone v3 0.01587]
    Q_0089 gold DOC_0022: graph rank 4, merged position 6, score 0.01562; context = D016[GRAPH+VECTOR g2 v1 0.03252]; D015[GRAPH g1 vNone 0.01639]; D026[VECTOR gNone v2 0.01613]; D001[GRAPH g3 vNone 0.01587]
```

Mechanism codes: A = a gold document is in neither route list (candidate generation); B = every missing gold document has graph rank <= 4 (RRF crowding); C = a missing gold document has graph rank > 4 (K cap); D = missing gold only in the vector list. Per-question lists, ranks and scores for all 30 hop3 questions of each seed (graph route with edge counts, vector route with distances, merged RRF ranking with scores) are in `part1_hop3_detail_seed7.txt`, `part1_hop3_detail_seed7mo.txt`, `part1_hop3_detail_seed11.txt`.

### Table 1.3 Can the graph route reach the gold documents? (engine `GraphTraverser`, stored graphs, recorded seeds)

```
seed 7 (kb_dev): graph vertices=82 edges=53
| group | N | recorded graph list covers all gold | T1 capped-20 directed | T2 uncapped directed | T3 undirected 3-hop | T4 undirected any distance | graph list covers gold but T1 does not (seed-direct/other chunks) |
|---|---|---|---|---|---|---|---|
| hop1 | 30 | 30 | 27 | 27 | 30 | 30 | 3 |
| hop2 | 30 | 18 | 18 | 18 | 18 | 24 | 0 |
| hop3 | 30 | 6 | 6 | 6 | 7 | 18 | 0 |
| alias | 15 | 9 | 9 | 9 | 9 | 10 | 0 |
  hop2: questions where the capped traversal returned exactly 20 paths: 0/30; median uncapped path count 2; max hop reached in the capped traversal: {1: 9, 2: 10, 0: 4, 3: 7}; max hop reached uncapped: {1: 9, 2: 10, 0: 4, 3: 7}
  hop3: questions where the capped traversal returned exactly 20 paths: 0/30; median uncapped path count 2; max hop reached in the capped traversal: {3: 11, 2: 8, 1: 9, 0: 2}; max hop reached uncapped: {3: 11, 2: 8, 1: 9, 0: 2}
  hop2: smallest hop limit at which ALL gold docs are covered (None = never within 12). undirected: {1: 18, 4: 2, 6: 2, 8: 2, None: 6}; directed (engine traversal, cap removed): {1: 18, None: 12}
  hop3: smallest hop limit at which ALL gold docs are covered (None = never within 12). undirected: {2: 6, 3: 1, 4: 1, 5: 3, 6: 2, 7: 2, 8: 1, 9: 1, 10: 1, None: 12}; directed (engine traversal, cap removed): {2: 6, None: 24}
```

```
seed 7 (kb_dev_merge_only): graph vertices=82 edges=55
| group | N | recorded graph list covers all gold | T1 capped-20 directed | T2 uncapped directed | T3 undirected 3-hop | T4 undirected any distance | graph list covers gold but T1 does not (seed-direct/other chunks) |
|---|---|---|---|---|---|---|---|
| hop1 | 30 | 30 | 26 | 26 | 30 | 30 | 4 |
| hop2 | 30 | 20 | 20 | 20 | 20 | 27 | 0 |
| hop3 | 30 | 10 | 10 | 10 | 11 | 20 | 0 |
| alias | 15 | 9 | 8 | 8 | 9 | 10 | 1 |
  hop2: questions where the capped traversal returned exactly 20 paths: 0/30; median uncapped path count 3; max hop reached in the capped traversal: {1: 8, 3: 11, 2: 6, 0: 5}; max hop reached uncapped: {1: 8, 3: 11, 2: 6, 0: 5}
  hop3: questions where the capped traversal returned exactly 20 paths: 0/30; median uncapped path count 4; max hop reached in the capped traversal: {3: 17, 2: 2, 1: 8, 0: 3}; max hop reached uncapped: {3: 17, 2: 2, 1: 8, 0: 3}
  hop2: smallest hop limit at which ALL gold docs are covered (None = never within 12). undirected: {1: 20, 4: 3, 6: 1, 12: 1, None: 5}; directed (engine traversal, cap removed): {1: 20, None: 10}
  hop3: smallest hop limit at which ALL gold docs are covered (None = never within 12). undirected: {2: 10, 3: 1, 4: 1, 5: 3, 6: 2, 7: 1, 9: 1, 11: 1, None: 10}; directed (engine traversal, cap removed): {2: 10, None: 20}
```

```
seed 11 (kb_seed11): graph vertices=79 edges=45
| group | N | recorded graph list covers all gold | T1 capped-20 directed | T2 uncapped directed | T3 undirected 3-hop | T4 undirected any distance | graph list covers gold but T1 does not (seed-direct/other chunks) |
|---|---|---|---|---|---|---|---|
| hop1 | 30 | 30 | 23 | 23 | 30 | 30 | 7 |
| hop2 | 30 | 15 | 15 | 15 | 16 | 17 | 0 |
| hop3 | 30 | 1 | 1 | 1 | 2 | 5 | 0 |
| alias | 15 | 10 | 9 | 9 | 10 | 10 | 1 |
  hop2: questions where the capped traversal returned exactly 20 paths: 0/30; median uncapped path count 1; max hop reached in the capped traversal: {0: 6, 2: 12, 1: 10, 3: 2}; max hop reached uncapped: {0: 6, 2: 12, 1: 10, 3: 2}
  hop3: questions where the capped traversal returned exactly 20 paths: 0/30; median uncapped path count 2; max hop reached in the capped traversal: {2: 20, 0: 8, 1: 2}; max hop reached uncapped: {2: 20, 0: 8, 1: 2}
  hop2: smallest hop limit at which ALL gold docs are covered (None = never within 12). undirected: {1: 16, 4: 1, None: 13}; directed (engine traversal, cap removed): {1: 15, None: 15}
  hop3: smallest hop limit at which ALL gold docs are covered (None = never within 12). undirected: {2: 1, 3: 1, 4: 1, 5: 2, None: 25}; directed (engine traversal, cap removed): {2: 1, None: 29}
```

T1 = engine behaviour (directed, 3 hops, 20 paths); T2 = cap removed; T3 = undirected within 3 hops; T4 = undirected, any distance (the longer-distance coverage passes through shared client/department hub vertices, see the hop-limit rows above). Gold-edge fate for the missing documents and path completeness: see `link_tables.txt` (L1, L2).

### Table 1.4 Gold-edge link: hop2/hop3 outcome by path completeness (all gold path edges are exact validated edges), and fate of the edge behind each unreached document

**seed 7 (kb_dev)**

```
L1 (seed 7 (kb_dev)): path completeness (all gold path edges have fate A or E) vs outcome
| group | path-complete? | N | hybrid ok | graph_only ok | graph top-4 ok | vector_only ok |
|---|---|---|---|---|---|---|
| hop2 | yes | 4 | 4 | 4 | 4 | 1 |
| hop2 | no | 26 | 14 | 14 | 14 | 4 |
| hop3 | yes | 1 | 0 | 1 | 1 | 0 |
| hop3 | no | 29 | 1 | 5 | 3 | 0 |
  hop3 gold path length distribution: {3: 30}
  hop3 questions by number of path edges with fate A/E: {0: 6, 1: 13, 2: 10, 3: 1} (key = edges with exact validated edge)
```

```
L2 (seed 7 (kb_dev)): hop3 questions, hybrid fails, a missing gold doc is in neither route list: fate of the gold edge supported by that doc
| gold edge fate | relation | count of (question, missing doc) |
|---|---|---|
| A validated, exact gold type | BUILT_WITH | 8 |
| A validated, exact gold type | SERVES | 6 |
| A validated, exact gold type | BELONGS_TO | 5 |
| D rejected for evidence reasons (not in the requested list) | DEVELOPS | 3 |
| B validated under another type / reversed, same endpoints | BELONGS_TO | 2 |
| C rejected by ontology | DEVELOPS | 2 |
| C rejected by ontology | MANAGES | 2 |
| D rejected for evidence reasons (not in the requested list) | BUILT_WITH | 2 |
| A validated, exact gold type | MANAGES | 1 |
| D rejected for evidence reasons (not in the requested list) | BELONGS_TO | 1 |
| D rejected for evidence reasons (not in the requested list) | SERVES | 1 |
  total (question, missing doc, edge) triples: 33; with fate A/E: 20
```

**seed 11 (kb_seed11)**

```
L1 (seed 11 (kb_seed11)): path completeness (all gold path edges have fate A or E) vs outcome
| group | path-complete? | N | hybrid ok | graph_only ok | graph top-4 ok | vector_only ok |
|---|---|---|---|---|---|---|
| hop2 | yes | 2 | 1 | 2 | 2 | 0 |
| hop2 | no | 28 | 10 | 13 | 13 | 4 |
| hop3 | yes | 0 | 0 | 0 | 0 | 0 |
| hop3 | no | 30 | 0 | 1 | 1 | 0 |
  hop3 gold path length distribution: {3: 30}
  hop3 questions by number of path edges with fate A/E: {0: 2, 1: 22, 2: 6} (key = edges with exact validated edge)
```

```
L2 (seed 11 (kb_seed11)): hop3 questions, hybrid fails, a missing gold doc is in neither route list: fate of the gold edge supported by that doc
| gold edge fate | relation | count of (question, missing doc) |
|---|---|---|
| A validated, exact gold type | SERVES | 10 |
| A validated, exact gold type | BELONGS_TO | 5 |
| D rejected for evidence reasons (not in the requested list) | BUILT_WITH | 5 |
| C rejected by ontology | DEVELOPS | 4 |
| C rejected by ontology | BUILT_WITH | 3 |
| D rejected for evidence reasons (not in the requested list) | SERVES | 3 |
| A validated, exact gold type | BUILT_WITH | 2 |
| A validated, exact gold type | DEVELOPS | 1 |
| C rejected by ontology | MANAGES | 1 |
| D rejected for evidence reasons (not in the requested list) | BELONGS_TO | 1 |
| D rejected for evidence reasons (not in the requested list) | DEVELOPS | 1 |
| D rejected for evidence reasons (not in the requested list) | MANAGES | 1 |
| F never extracted | BELONGS_TO | 1 |
| F never extracted | SERVES | 1 |
  total (question, missing doc, edge) triples: 39; with fate A/E: 18
```

### Table 1.5 EXPLORATORY, NOT A FIX, SEED ALREADY USED: counterfactual re-ranking of the recorded lists (no change to src)

**seed 7 (kb_dev)**

| selection rule | hop1 | hop2 | hop3 | alias | hop3 gained vs baseline | hop3 lost vs baseline | other-group lost vs baseline |
|---|---|---|---|---|---|---|---|
| baseline RRF top-4 (replica) | 30/30 = 100.0% [88.6, 100.0] | 18/30 = 60.0% [42.3, 75.4] | 1/30 = 3.3% [0.6, 16.7] | 10/15 = 66.7% [41.7, 84.8] | 0 | 0 | 0 |
| graph top-4 only (K matched) | 30/30 = 100.0% [88.6, 100.0] | 18/30 = 60.0% [42.3, 75.4] | 4/30 = 13.3% [5.3, 29.7] | 9/15 = 60.0% [35.7, 80.2] | 3 | 0 | 5 |
| graph top-3 + vector top-1 | 30/30 = 100.0% [88.6, 100.0] | 17/30 = 56.7% [39.2, 72.6] | 1/30 = 3.3% [0.6, 16.7] | 11/15 = 73.3% [48.0, 89.1] | 0 | 0 | 2 |
| graph top-2 + vector top-2 | 30/30 = 100.0% [88.6, 100.0] | 8/30 = 26.7% [14.2, 44.4] | 0/30 = 0.0% [0.0, 11.4] | 8/15 = 53.3% [30.1, 75.2] | 0 | 1 | 12 |
| weighted RRF w_graph=2 (K=4) | 30/30 = 100.0% [88.6, 100.0] | 19/30 = 63.3% [45.5, 78.1] | 4/30 = 13.3% [5.3, 29.7] | 11/15 = 73.3% [48.0, 89.1] | 3 | 0 | 0 |
| weighted RRF w_graph=3 (K=4) | 30/30 = 100.0% [88.6, 100.0] | 19/30 = 63.3% [45.5, 78.1] | 4/30 = 13.3% [5.3, 29.7] | 11/15 = 73.3% [48.0, 89.1] | 3 | 0 | 0 |
| weighted RRF w_graph=5 (K=4) | 30/30 = 100.0% [88.6, 100.0] | 19/30 = 63.3% [45.5, 78.1] | 4/30 = 13.3% [5.3, 29.7] | 11/15 = 73.3% [48.0, 89.1] | 3 | 0 | 0 |
| weighted RRF w_graph=10 (K=4) | 30/30 = 100.0% [88.6, 100.0] | 19/30 = 63.3% [45.5, 78.1] | 5/30 = 16.7% [7.3, 33.6] | 11/15 = 73.3% [48.0, 89.1] | 4 | 0 | 0 |
| RRF top-6 (K=6) | 30/30 = 100.0% [88.6, 100.0] | 20/30 = 66.7% [48.8, 80.8] | 5/30 = 16.7% [7.3, 33.6] | 12/15 = 80.0% [54.8, 93.0] | 4 | 0 | 0 |
| RRF top-8 (K=8) | 30/30 = 100.0% [88.6, 100.0] | 20/30 = 66.7% [48.8, 80.8] | 6/30 = 20.0% [9.5, 37.3] | 12/15 = 80.0% [54.8, 93.0] | 5 | 0 | 0 |
| graph full list as evaluated in graph_only (K variable) | 30/30 = 100.0% [88.6, 100.0] | 18/30 = 60.0% [42.3, 75.4] | 6/30 = 20.0% [9.5, 37.3] | 9/15 = 60.0% [35.7, 80.2] | 5 | 0 | 5 |

**seed 7 merge-only KB**

| selection rule | hop1 | hop2 | hop3 | alias | hop3 gained vs baseline | hop3 lost vs baseline | other-group lost vs baseline |
|---|---|---|---|---|---|---|---|
| baseline RRF top-4 (replica) | 30/30 = 100.0% [88.6, 100.0] | 16/30 = 53.3% [36.1, 69.8] | 1/30 = 3.3% [0.6, 16.7] | 10/15 = 66.7% [41.7, 84.8] | 0 | 0 | 0 |
| graph top-4 only (K matched) | 30/30 = 100.0% [88.6, 100.0] | 18/30 = 60.0% [42.3, 75.4] | 4/30 = 13.3% [5.3, 29.7] | 9/15 = 60.0% [35.7, 80.2] | 3 | 0 | 5 |
| graph top-3 + vector top-1 | 30/30 = 100.0% [88.6, 100.0] | 16/30 = 53.3% [36.1, 69.8] | 1/30 = 3.3% [0.6, 16.7] | 11/15 = 73.3% [48.0, 89.1] | 0 | 0 | 2 |
| graph top-2 + vector top-2 | 30/30 = 100.0% [88.6, 100.0] | 8/30 = 26.7% [14.2, 44.4] | 0/30 = 0.0% [0.0, 11.4] | 7/15 = 46.7% [24.8, 69.9] | 0 | 1 | 11 |
| weighted RRF w_graph=2 (K=4) | 30/30 = 100.0% [88.6, 100.0] | 18/30 = 60.0% [42.3, 75.4] | 4/30 = 13.3% [5.3, 29.7] | 11/15 = 73.3% [48.0, 89.1] | 3 | 0 | 0 |
| weighted RRF w_graph=3 (K=4) | 30/30 = 100.0% [88.6, 100.0] | 18/30 = 60.0% [42.3, 75.4] | 4/30 = 13.3% [5.3, 29.7] | 11/15 = 73.3% [48.0, 89.1] | 3 | 0 | 0 |
| weighted RRF w_graph=5 (K=4) | 30/30 = 100.0% [88.6, 100.0] | 18/30 = 60.0% [42.3, 75.4] | 4/30 = 13.3% [5.3, 29.7] | 11/15 = 73.3% [48.0, 89.1] | 3 | 0 | 0 |
| weighted RRF w_graph=10 (K=4) | 30/30 = 100.0% [88.6, 100.0] | 18/30 = 60.0% [42.3, 75.4] | 5/30 = 16.7% [7.3, 33.6] | 11/15 = 73.3% [48.0, 89.1] | 4 | 0 | 0 |
| RRF top-6 (K=6) | 30/30 = 100.0% [88.6, 100.0] | 20/30 = 66.7% [48.8, 80.8] | 5/30 = 16.7% [7.3, 33.6] | 12/15 = 80.0% [54.8, 93.0] | 4 | 0 | 0 |
| RRF top-8 (K=8) | 30/30 = 100.0% [88.6, 100.0] | 21/30 = 70.0% [52.1, 83.3] | 7/30 = 23.3% [11.8, 40.9] | 12/15 = 80.0% [54.8, 93.0] | 6 | 0 | 0 |
| graph full list as evaluated in graph_only (K variable) | 30/30 = 100.0% [88.6, 100.0] | 20/30 = 66.7% [48.8, 80.8] | 10/30 = 33.3% [19.2, 51.2] | 9/15 = 60.0% [35.7, 80.2] | 9 | 0 | 5 |

**seed 11**

| selection rule | hop1 | hop2 | hop3 | alias | hop3 gained vs baseline | hop3 lost vs baseline | other-group lost vs baseline |
|---|---|---|---|---|---|---|---|
| baseline RRF top-4 (replica) | 30/30 = 100.0% [88.6, 100.0] | 11/30 = 36.7% [21.9, 54.5] | 0/30 = 0.0% [0.0, 11.4] | 12/15 = 80.0% [54.8, 93.0] | 0 | 0 | 0 |
| graph top-4 only (K matched) | 30/30 = 100.0% [88.6, 100.0] | 15/30 = 50.0% [33.2, 66.8] | 1/30 = 3.3% [0.6, 16.7] | 10/15 = 66.7% [41.7, 84.8] | 1 | 0 | 2 |
| graph top-3 + vector top-1 | 30/30 = 100.0% [88.6, 100.0] | 11/30 = 36.7% [21.9, 54.5] | 0/30 = 0.0% [0.0, 11.4] | 11/15 = 73.3% [48.0, 89.1] | 0 | 0 | 2 |
| graph top-2 + vector top-2 | 30/30 = 100.0% [88.6, 100.0] | 7/30 = 23.3% [11.8, 40.9] | 0/30 = 0.0% [0.0, 11.4] | 8/15 = 53.3% [30.1, 75.2] | 0 | 0 | 8 |
| weighted RRF w_graph=2 (K=4) | 30/30 = 100.0% [88.6, 100.0] | 13/30 = 43.3% [27.4, 60.8] | 1/30 = 3.3% [0.6, 16.7] | 12/15 = 80.0% [54.8, 93.0] | 1 | 0 | 0 |
| weighted RRF w_graph=3 (K=4) | 30/30 = 100.0% [88.6, 100.0] | 13/30 = 43.3% [27.4, 60.8] | 1/30 = 3.3% [0.6, 16.7] | 12/15 = 80.0% [54.8, 93.0] | 1 | 0 | 0 |
| weighted RRF w_graph=5 (K=4) | 30/30 = 100.0% [88.6, 100.0] | 13/30 = 43.3% [27.4, 60.8] | 1/30 = 3.3% [0.6, 16.7] | 12/15 = 80.0% [54.8, 93.0] | 1 | 0 | 0 |
| weighted RRF w_graph=10 (K=4) | 30/30 = 100.0% [88.6, 100.0] | 14/30 = 46.7% [30.2, 63.9] | 1/30 = 3.3% [0.6, 16.7] | 12/15 = 80.0% [54.8, 93.0] | 1 | 0 | 0 |
| RRF top-6 (K=6) | 30/30 = 100.0% [88.6, 100.0] | 15/30 = 50.0% [33.2, 66.8] | 2/30 = 6.7% [1.8, 21.3] | 12/15 = 80.0% [54.8, 93.0] | 2 | 0 | 0 |
| RRF top-8 (K=8) | 30/30 = 100.0% [88.6, 100.0] | 15/30 = 50.0% [33.2, 66.8] | 2/30 = 6.7% [1.8, 21.3] | 12/15 = 80.0% [54.8, 93.0] | 2 | 0 | 0 |
| graph full list as evaluated in graph_only (K variable) | 30/30 = 100.0% [88.6, 100.0] | 15/30 = 50.0% [33.2, 66.8] | 1/30 = 3.3% [0.6, 16.7] | 10/15 = 66.7% [41.7, 84.8] | 1 | 0 | 2 |

Reading: weighting the graph route by 2 (equal results for weights 2, 3 and 5) lifts hop3 from 1 to 4 (seed 7), 1 to 4 (merge-only) and 0 to 1 (seed 11), hop2 from 18 to 19 and 11 to 13, alias from 10 to 11 (seed 7) and 12 to 12 (seed 11), without losing a question that the baseline answered (questions lost on seed 7 / merge-only / seed 11: 0 / 0 / 0). Hop3 questions gained at weight 2: seed 7 ['Q_0066', 'Q_0067', 'Q_0089'], seed 11 ['Q_0073']; hop2 questions gained: seed 7 ['Q_0045'], seed 11 ['Q_0034', 'Q_0036']. Most of the recoverable hop3 mass is not reachable by re-ranking at K=4: even the union of both lists covers all gold documents for 6/30 (seed 7) and 2/30 (seed 11) hop3 questions.

### What this does NOT show (Part 1)

- It does not show that weighted RRF (or any re-ranking) is a fix. The weights were evaluated on the same questions and seeds that revealed the problem; the plateau between weights 2 and 5 is an observation on 60 hop2/hop3 questions per seed.
- It does not show how a graph-biased ranking affects questions that vector retrieval answers better; hop1 is already saturated in every mode, and the unanswerable group is not part of this retrieval-only metric.
- It does not explain, edge by edge, why each of the unreached gold documents is unreachable. For the questions in mechanism A it shows that the stored graph (not the 20-path cap, not edge direction) limits the traversal and that complete exact paths are rare; in many cases the document's own edge is an exact validated edge (Table 1.4, L2) but the upstream part of the chain is missing or sits on a different component. A per-edge chain audit was not done.
- It does not show that a larger hop limit helps. `retrieval.max_traversal_hops` is defined in `configs/default.yaml` and `src/core/configuration.py:105` but is not passed to `GraphTraverser` (`GraphEvidenceRetriever` constructs `GraphTraverser(max_candidates=...)`, default 3 hops; grep over `src/` finds no other use), so the config key currently has no effect. The undirected any-distance coverage (T4) is reached through hub vertices and would flood a 4-chunk context; no retrieval was run with a larger limit.
- It does not measure answer quality; the metric is whether all gold documents are among the returned chunks.
- The planner's plans are cached (one generation per question); planner variance is not examined. The graph route has no score, only a rank; vector distances and RRF scores are recorded, graph scores are not defined by the engine.
- The comparison in FINAL_REPORT.md between graph_only (variable K) and hybrid (K=4) was descriptive, not one of the pre-registered tests (P1-P3 compare hybrid with graph-free baselines at K=4), so the pre-registered decisions are unaffected; descriptive graph_only numbers should be reported at K=4 as well.

## Part 2. Why are 11 of 15 gold DEVELOPS edges rejected by the ontology on seed 11?

### Finding

The cause is (d), something else: **the extractor assigns the right relation type and the right entity types but reverses the direction.** On seed 11, 11 of the 15 gold DEVELOPS edges (fate C, rejected by ontology) were extracted as `PRODUCT -DEVELOPS-> DEPARTMENT`, while the sentence in the document is active with the department as subject (for example "X designs and engineers Y"). The ontology rule for DEVELOPS (`configs/domains/company.yaml`: source EMPLOYEE or DEPARTMENT, target PRODUCT) rejects the reversed pair, as it should. The gold type pair of all 15 gold DEVELOPS edges is DEPARTMENT -> PRODUCT, which the rule allows (15/15), so (b) the ontology being too narrow is excluded; the mention types assigned by the extractor equal the gold types for 15/15 edges, so (a) mention-type errors are excluded for these edges; and (c) revalidation could not help because the canonical types equal the mention types (revalidation did evaluate these triples, they are eligible, and left them rejected, see Table C). The remaining seed-11 DEVELOPS edges: 2 validated exactly (fate A) and 2 extracted in the right direction but rejected for evidence reasons (one endpoint-not-in-evidence, one quote mismatch). Seed 7 shows the same pattern: 12 of 15 gold DEVELOPS edges are reversed (of which 7 are rejected by the ontology and 5 are reversed and also rejected for evidence reasons), 3 validated exactly. Over both seeds 23 of 30 extracted DEVELOPS triples joining the endpoints of a gold DEVELOPS edge are reversed, against MANAGES 5/30, BELONGS_TO 1/15, BUILT_WITH 5/21, SERVES 0/45 for the other relations (same-relation triples only). The direction depends on the sentence template in these corpora: "spearheads the technical development of" is extracted in the gold direction 6 of 7 times, the other three templates ('is directly responsible for developing', 'builds and maintains', 'designs and engineers') only 1 of 23 times. The prompt already states the direction twice (the constraint line `(Allowed: ['EMPLOYEE', 'DEPARTMENT'] -> ['PRODUCT'])` built in `src/extraction/relation_extractor.py:154` and instruction 7 of `prompts/relationship_extraction.txt`), so this is gemma3:4b not following the stated direction for these verbs, not an engine or ontology defect. Because the reversed pair is the only thing wrong, a post-extraction check that uses the ontology alone could repair it (candidate fix 1).

### Table 2.1 Per gold DEVELOPS edge, seed 11

ontology rule DEVELOPS: source in ['EMPLOYEE', 'DEPARTMENT'], target in ['PRODUCT']
gold DEVELOPS edges: 15; gold (source type, target type) pairs: {('DEPARTMENT', 'PRODUCT'): 15}; pairs allowed by the DEVELOPS rule: 15/15

TABLE A (seed 11 (kb_seed11)): per gold DEVELOPS edge
| edge | doc | gold source (type) | gold target (type) | fate | mention types source / target | extracted relation (dir, mention types, status ML -> final) | cause |
|---|---|---|---|---|---|---|---|
| EDGE_0001 | DOC_0022 | Raelonmar Infrastructure Group (DEPARTMENT) | Lorelrix Mesh (PRODUCT) | C | ['DEPARTMENT'] / ['PRODUCT'] | DEVELOPS REV (PRODUCT->DEPARTMENT) REJECTED_ONTOLOGY->REJECTED_ONTOLOGY | REVERSED |
| EDGE_0002 | DOC_0023 | Maelinvane Core Platform Division (DEPARTMENT) | Eldenfen Matrix (PRODUCT) | C | ['DEPARTMENT'] / ['PRODUCT'] | DEVELOPS REV (PRODUCT->DEPARTMENT) REJECTED_ONTOLOGY->REJECTED_ONTOLOGY | REVERSED |
| EDGE_0003 | DOC_0024 | Aethorvath Applied Automation Wing (DEPARTMENT) | Zorarvek Cluster (PRODUCT) | C | ['DEPARTMENT'] / ['PRODUCT'] | DEVELOPS REV (PRODUCT->DEPARTMENT) REJECTED_ONTOLOGY->REJECTED_ONTOLOGY | REVERSED |
| EDGE_0004 | DOC_0025 | Morithvor Foundational Services Unit (DEPARTMENT) | Synaldor Hub (PRODUCT) | A | ['DEPARTMENT'] / ['PRODUCT'] | DEVELOPS fwd (DEPARTMENT->PRODUCT) VALIDATED->VALIDATED | OK (validated exact) |
| EDGE_0005 | DOC_0026 | Zantothvane Applied Automation Wing (DEPARTMENT) | Sarkantor Engine (PRODUCT) | A | ['DEPARTMENT'] / ['PRODUCT'] | DEVELOPS fwd (DEPARTMENT->PRODUCT) VALIDATED->VALIDATED | OK (validated exact) |
| EDGE_0006 | DOC_0027 | Fennuthreth Systems Engineering Unit (DEPARTMENT) | Synokwyn Protocol (PRODUCT) | C | ['DEPARTMENT'] / ['PRODUCT'] | DEVELOPS REV (PRODUCT->DEPARTMENT) REJECTED_ONTOLOGY->REJECTED_ONTOLOGY | REVERSED |
| EDGE_0007 | DOC_0028 | Ulxintor Systems Engineering Unit (DEPARTMENT) | Vaelelmar Hub (PRODUCT) | C | ['DEPARTMENT'] / ['PRODUCT'] | DEVELOPS REV (PRODUCT->DEPARTMENT) REJECTED_ONTOLOGY->REJECTED_ONTOLOGY | REVERSED |
| EDGE_0008 | DOC_0029 | Eldismir Applied Automation Wing (DEPARTMENT) | Vaeluntex Protocol (PRODUCT) | C | ['DEPARTMENT'] / ['PRODUCT'] | DEVELOPS REV (PRODUCT->DEPARTMENT) REJECTED_ONTOLOGY->REJECTED_ONTOLOGY | REVERSED |
| EDGE_0009 | DOC_0030 | Noxinzor Systems Engineering Unit (DEPARTMENT) | Wyrinzar Workspace (PRODUCT) | C | ['DEPARTMENT'] / ['PRODUCT'] | DEVELOPS REV (PRODUCT->DEPARTMENT) REJECTED_ONTOLOGY->REJECTED_ONTOLOGY | REVERSED |
| EDGE_0010 | DOC_0031 | Gorardor Analytics Laboratory (DEPARTMENT) | Syniktex Mesh (PRODUCT) | C | ['DEPARTMENT'] / ['PRODUCT'] | DEVELOPS REV (PRODUCT->DEPARTMENT) REJECTED_ONTOLOGY->REJECTED_ONTOLOGY | REVERSED |
| EDGE_0011 | DOC_0032 | Aldokrak Core Platform Division (DEPARTMENT) | Fennosron Relay (PRODUCT) | C | ['DEPARTMENT'] / ['PRODUCT'] | DEVELOPS REV (PRODUCT->DEPARTMENT) REJECTED_ONTOLOGY->REJECTED_ONTOLOGY | REVERSED |
| EDGE_0012 | DOC_0033 | Zorathmoth Analytics Laboratory (DEPARTMENT) | Quenontis Relay (PRODUCT) | C | ['DEPARTMENT'] / ['PRODUCT'] | DEVELOPS REV (PRODUCT->DEPARTMENT) REJECTED_ONTOLOGY->REJECTED_ONTOLOGY | REVERSED |
| EDGE_0013 | DOC_0034 | Haelostor Core Platform Division (DEPARTMENT) | Lorelrix Mesh (PRODUCT) | D | ['DEPARTMENT'] / ['PRODUCT'] | DEVELOPS fwd (DEPARTMENT->PRODUCT) REJECTED_ENDPOINT_NOT_IN_EVIDENCE->REJECTED_ENDPOINT_NOT_IN_EVIDENCE | EVIDENCE_REJECT |
| EDGE_0014 | DOC_0035 | Xylounmar Infrastructure Group (DEPARTMENT) | Eldenfen Matrix (PRODUCT) | D | ['DEPARTMENT'] / ['PRODUCT'] | DEVELOPS fwd (DEPARTMENT->PRODUCT) REJECTED_QUOTE_MISMATCH->REJECTED_QUOTE_MISMATCH | EVIDENCE_REJECT |
| EDGE_0015 | DOC_0036 | Vandolkest Systems Engineering Unit (DEPARTMENT) | Zorarvek Cluster (PRODUCT) | C | ['DEPARTMENT'] / ['PRODUCT'] | DEVELOPS REV (PRODUCT->DEPARTMENT) REJECTED_ONTOLOGY->REJECTED_ONTOLOGY | REVERSED |

CAUSE COUNTS (seed 11 (kb_seed11)): {'REVERSED': 11, 'OK (validated exact)': 2, 'EVIDENCE_REJECT': 2}
FATE COUNTS (seed 11 (kb_seed11)): {'C': 11, 'A': 2, 'D': 2}

### Table 2.2 Per gold DEVELOPS edge, seed 7 (kb_dev)

TABLE A (seed 7 (kb_dev, after fixes)): per gold DEVELOPS edge
| edge | doc | gold source (type) | gold target (type) | fate | mention types source / target | extracted relation (dir, mention types, status ML -> final) | cause |
|---|---|---|---|---|---|---|---|
| EDGE_0001 | DOC_0022 | Zanistis Infrastructure Group (DEPARTMENT) | Haelentor Hub (PRODUCT) | C | ['DEPARTMENT'] / ['PRODUCT'] | DEVELOPS REV (PRODUCT->DEPARTMENT) REJECTED_ONTOLOGY->REJECTED_ONTOLOGY | REVERSED |
| EDGE_0002 | DOC_0023 | Zantonnax Infrastructure Group (DEPARTMENT) | Iskardyn Pipeline (PRODUCT) | C | ['DEPARTMENT'] / ['PRODUCT'] | DEVELOPS REV (PRODUCT->DEPARTMENT) REJECTED_ONTOLOGY->REJECTED_ONTOLOGY | REVERSED |
| EDGE_0003 | DOC_0024 | Oryxonvath Systems Engineering Unit (DEPARTMENT) | Sarkinfen Matrix (PRODUCT) | D | ['DEPARTMENT'] / ['PRODUCT'] | DEVELOPS REV (PRODUCT->DEPARTMENT) REJECTED_ENDPOINT_NOT_IN_EVIDENCE->REJECTED_ENDPOINT_NOT_IN_EVIDENCE | REVERSED |
| EDGE_0004 | DOC_0025 | Synethgath Systems Engineering Unit (DEPARTMENT) | Haelenrix Gateway (PRODUCT) | C | ['DEPARTMENT'] / ['PRODUCT'] | DEVELOPS REV (PRODUCT->DEPARTMENT) REJECTED_ONTOLOGY->REJECTED_ONTOLOGY | REVERSED |
| EDGE_0005 | DOC_0026 | Kronanvek Core Platform Division (DEPARTMENT) | Wyrithwyn Pipeline (PRODUCT) | A | ['DEPARTMENT'] / ['PRODUCT'] | DEVELOPS fwd (DEPARTMENT->PRODUCT) VALIDATED->VALIDATED | OK (validated exact) |
| EDGE_0006 | DOC_0027 | Aetharkest Systems Engineering Unit (DEPARTMENT) | Zanolric Cluster (PRODUCT) | D | ['DEPARTMENT'] / ['PRODUCT'] | DEVELOPS REV (PRODUCT->DEPARTMENT) REJECTED_ENDPOINT_NOT_IN_EVIDENCE->REJECTED_ENDPOINT_NOT_IN_EVIDENCE | REVERSED |
| EDGE_0007 | DOC_0028 | Iskorlan Applied Automation Wing (DEPARTMENT) | Jorustex Nexus (PRODUCT) | A | ['DEPARTMENT'] / ['PRODUCT'] | DEVELOPS fwd (DEPARTMENT->PRODUCT) VALIDATED->VALIDATED | OK (validated exact) |
| EDGE_0008 | DOC_0029 | Kaelinlux Systems Engineering Unit (DEPARTMENT) | Lyronvor Nexus (PRODUCT) | C | ['DEPARTMENT'] / ['PRODUCT'] | DEVELOPS REV (PRODUCT->DEPARTMENT) REJECTED_ONTOLOGY->REJECTED_ONTOLOGY | REVERSED |
| EDGE_0009 | DOC_0030 | Noxontor Foundational Services Unit (DEPARTMENT) | Faelolnok Protocol (PRODUCT) | D | ['DEPARTMENT'] / ['PRODUCT'] | DEVELOPS REV (PRODUCT->DEPARTMENT) REJECTED_ENDPOINT_NOT_IN_EVIDENCE->REJECTED_ENDPOINT_NOT_IN_EVIDENCE | REVERSED |
| EDGE_0010 | DOC_0031 | Morenvek Analytics Laboratory (DEPARTMENT) | Ulxarron Matrix (PRODUCT) | A | ['DEPARTMENT'] / ['PRODUCT'] | DEVELOPS fwd (DEPARTMENT->PRODUCT) VALIDATED->VALIDATED | OK (validated exact) |
| EDGE_0011 | DOC_0032 | Eldornax Infrastructure Group (DEPARTMENT) | Eldothvix Hub (PRODUCT) | D | ['DEPARTMENT'] / ['PRODUCT'] | DEVELOPS REV (PRODUCT->DEPARTMENT) REJECTED_QUOTE_MISMATCH->REJECTED_QUOTE_MISMATCH | REVERSED |
| EDGE_0012 | DOC_0033 | Durnalnok Analytics Laboratory (DEPARTMENT) | Maelengath Hub (PRODUCT) | C | ['DEPARTMENT'] / ['PRODUCT'] | DEVELOPS REV (PRODUCT->DEPARTMENT) REJECTED_ONTOLOGY->REJECTED_ONTOLOGY | REVERSED |
| EDGE_0013 | DOC_0034 | Nebunnax Foundational Services Unit (DEPARTMENT) | Haelentor Hub (PRODUCT) | C | ['DEPARTMENT'] / ['PRODUCT'] | DEVELOPS REV (PRODUCT->DEPARTMENT) REJECTED_ONTOLOGY->REJECTED_ONTOLOGY | REVERSED |
| EDGE_0014 | DOC_0035 | Vandothnok Foundational Services Unit (DEPARTMENT) | Iskardyn Pipeline (PRODUCT) | D | ['DEPARTMENT'] / ['PRODUCT'] | DEVELOPS REV (PRODUCT->DEPARTMENT) REJECTED_QUOTE_MISMATCH->REJECTED_QUOTE_MISMATCH | REVERSED |
| EDGE_0015 | DOC_0036 | Brecolron Systems Engineering Unit (DEPARTMENT) | Sarkinfen Matrix (PRODUCT) | C | ['DEPARTMENT'] / ['PRODUCT'] | DEVELOPS REV (PRODUCT->DEPARTMENT) REJECTED_ONTOLOGY->REJECTED_ONTOLOGY | REVERSED |

CAUSE COUNTS (seed 7 (kb_dev, after fixes)): {'REVERSED': 12, 'OK (validated exact)': 3}
FATE COUNTS (seed 7 (kb_dev, after fixes)): {'C': 7, 'D': 5, 'A': 3}

### Table 2.3 Extracted type pairs vs the pairs the ontology allows

**Seed 11**

TABLE B (seed 11 (kb_seed11)): extracted (relation, source mention type -> target mention type) for triples joining the endpoints of a gold DEVELOPS edge
| relation | direction vs gold | source mention type | target mention type | pair allowed by that relation's rule | status at mention level | final status | count |
|---|---|---|---|---|---|---|---|
| DEVELOPS | REV | PRODUCT | DEPARTMENT | False | REJECTED_ONTOLOGY | REJECTED_ONTOLOGY | 11 |
| DEVELOPS | fwd | DEPARTMENT | PRODUCT | True | VALIDATED | VALIDATED | 2 |
| DEVELOPS | fwd | DEPARTMENT | PRODUCT | True | REJECTED_ENDPOINT_NOT_IN_EVIDENCE | REJECTED_ENDPOINT_NOT_IN_EVIDENCE | 1 |
| DEVELOPS | fwd | DEPARTMENT | PRODUCT | True | REJECTED_QUOTE_MISMATCH | REJECTED_QUOTE_MISMATCH | 1 |
allowed pairs per relation in configs/domains/company.yaml: MANAGES: ['EMPLOYEE'] -> ['EMPLOYEE', 'DEPARTMENT', 'PRODUCT']; BELONGS_TO: ['EMPLOYEE', 'PRODUCT'] -> ['DEPARTMENT']; DEVELOPS: ['EMPLOYEE', 'DEPARTMENT'] -> ['PRODUCT']; BUILT_WITH: ['PRODUCT'] -> ['TECHNOLOGY']; SERVES: ['PRODUCT', 'DEPARTMENT'] -> ['CLIENT']

**Seed 7 (kb_dev)**

TABLE B (seed 7 (kb_dev, after fixes)): extracted (relation, source mention type -> target mention type) for triples joining the endpoints of a gold DEVELOPS edge
| relation | direction vs gold | source mention type | target mention type | pair allowed by that relation's rule | status at mention level | final status | count |
|---|---|---|---|---|---|---|---|
| DEVELOPS | REV | PRODUCT | DEPARTMENT | False | REJECTED_ONTOLOGY | REJECTED_ONTOLOGY | 7 |
| DEVELOPS | REV | PRODUCT | DEPARTMENT | False | REJECTED_ENDPOINT_NOT_IN_EVIDENCE | REJECTED_ENDPOINT_NOT_IN_EVIDENCE | 3 |
| DEVELOPS | fwd | DEPARTMENT | PRODUCT | True | VALIDATED | VALIDATED | 3 |
| DEVELOPS | REV | PRODUCT | DEPARTMENT | False | REJECTED_QUOTE_MISMATCH | REJECTED_QUOTE_MISMATCH | 2 |
allowed pairs per relation in configs/domains/company.yaml: MANAGES: ['EMPLOYEE'] -> ['EMPLOYEE', 'DEPARTMENT', 'PRODUCT']; BELONGS_TO: ['EMPLOYEE', 'PRODUCT'] -> ['DEPARTMENT']; DEVELOPS: ['EMPLOYEE', 'DEPARTMENT'] -> ['PRODUCT']; BUILT_WITH: ['PRODUCT'] -> ['TECHNOLOGY']; SERVES: ['PRODUCT', 'DEPARTMENT'] -> ['CLIENT']

### Table 2.4 All triples rejected by the ontology (every relation): mention types vs canonical types after resolution

**Seed 11**

TABLE C (seed 11 (kb_seed11)): ALL triples whose mention-level status is REJECTED_ONTOLOGY: relation, mention type pair, canonical type pair, final status
| relation | mention types | canonical types | canonical pair allowed | final status | count |
|---|---|---|---|---|---|
| DEVELOPS | PRODUCT->DEPARTMENT | PRODUCT->DEPARTMENT | canonical pair allowed=False | REJECTED_ONTOLOGY | 11 |
| SERVES | EMPLOYEE->CLIENT | EMPLOYEE->EMPLOYEE | canonical pair allowed=False | REJECTED_ONTOLOGY | 3 |
| MANAGES | DEPARTMENT->EMPLOYEE | DEPARTMENT->EMPLOYEE | canonical pair allowed=False | REJECTED_ONTOLOGY | 2 |
| DEVELOPS | EMPLOYEE->DEPARTMENT | EMPLOYEE->DEPARTMENT | canonical pair allowed=False | REJECTED_ONTOLOGY | 1 |
| SERVES | EMPLOYEE->DEPARTMENT | EMPLOYEE->DEPARTMENT | canonical pair allowed=False | REJECTED_ONTOLOGY | 1 |
| MANAGES | PRODUCT->EMPLOYEE | EMPLOYEE->EMPLOYEE | canonical pair allowed=True | VALIDATED | 1 |
| SERVES | DEPARTMENT->PRODUCT | DEPARTMENT->CLIENT | canonical pair allowed=True | VALIDATED | 1 |
| BUILT_WITH | TECHNOLOGY->PRODUCT | PRODUCT->PRODUCT | canonical pair allowed=False | REJECTED_ONTOLOGY | 1 |
| DEVELOPS | PRODUCT->TECHNOLOGY | PRODUCT->PRODUCT | canonical pair allowed=False | REJECTED_ONTOLOGY | 1 |
| BUILT_WITH | TECHNOLOGY->PRODUCT | TECHNOLOGY->PRODUCT | canonical pair allowed=False | REJECTED_ONTOLOGY | 1 |
| MAINTAINED | PRODUCT->DEPARTMENT | PRODUCT->DEPARTMENT | canonical pair allowed=False | REJECTED_ONTOLOGY | 1 |
REJECTED_ONTOLOGY at mention level: 24; canonical pair would be allowed: 2; final status VALIDATED (revived by revalidation): 2; still rejected: 22
resolution_report.json revalidation summary: recomputed=69 validated_to_rejected=2 rejected_to_validated=2

**Seed 7 (kb_dev)**

TABLE C (seed 7 (kb_dev, after fixes)): ALL triples whose mention-level status is REJECTED_ONTOLOGY: relation, mention type pair, canonical type pair, final status
| relation | mention types | canonical types | canonical pair allowed | final status | count |
|---|---|---|---|---|---|
| DEVELOPS | PRODUCT->DEPARTMENT | PRODUCT->DEPARTMENT | canonical pair allowed=False | REJECTED_ONTOLOGY | 7 |
| MANAGES | EMPLOYEE->CLIENT | EMPLOYEE->DEPARTMENT | canonical pair allowed=True | VALIDATED | 1 |
| MANAGES | DEPARTMENT->EMPLOYEE | DEPARTMENT->EMPLOYEE | canonical pair allowed=False | REJECTED_ONTOLOGY | 1 |
| BELONGS_TO | DEPARTMENT->EMPLOYEE | DEPARTMENT->EMPLOYEE | canonical pair allowed=False | REJECTED_ONTOLOGY | 1 |
| SERVES | EMPLOYEE->CLIENT | EMPLOYEE->CLIENT | canonical pair allowed=False | REJECTED_ONTOLOGY | 1 |
| SERVES | EMPLOYEE->CLIENT | EMPLOYEE->EMPLOYEE | canonical pair allowed=False | REJECTED_ONTOLOGY | 1 |
| BUILT_WITH | TECHNOLOGY->PRODUCT | TECHNOLOGY->PRODUCT | canonical pair allowed=False | REJECTED_ONTOLOGY | 1 |
| MAINTAINES | DEPARTMENT->PRODUCT | DEPARTMENT->TECHNOLOGY | canonical pair allowed=False | REJECTED_ONTOLOGY | 1 |
| MAINTAINS | DEPARTMENT->PRODUCT | DEPARTMENT->TECHNOLOGY | canonical pair allowed=False | REJECTED_ONTOLOGY | 1 |
REJECTED_ONTOLOGY at mention level: 15; canonical pair would be allowed: 1; final status VALIDATED (revived by revalidation): 1; still rejected: 14
resolution_report.json revalidation summary: recomputed=70 validated_to_rejected=3 rejected_to_validated=1

### Table 2.5 Direction of extracted triples vs gold, by gold relation, and DEVELOPS by sentence template

**seed 7 (kb_dev)**

L3 (seed 7 (kb_dev)): extracted triples joining the two endpoints of a gold edge (in the supporting doc): direction vs gold
| gold relation | extracted relation | direction | count |
|---|---|---|---|
| BELONGS_TO | extracted MANAGES | fwd | 15 |
| BELONGS_TO | same relation | fwd | 7 |
| BELONGS_TO | extracted MANAGES | REV | 1 |
| BELONGS_TO | same relation | REV | 1 |
| BELONGS_TO | no triple | - | 1 |
| BUILT_WITH | same relation | fwd | 10 |
| BUILT_WITH | same relation | REV | 2 |
| DEVELOPS | same relation | REV | 12 |
| DEVELOPS | same relation | fwd | 3 |
| MANAGES | same relation | fwd | 14 |
| MANAGES | extracted DEVELOPS | fwd | 4 |
| MANAGES | extracted BELONGS_TO | fwd | 2 |
| MANAGES | same relation | REV | 2 |
| MANAGES | extracted BELONGS_TO | REV | 1 |
| MANAGES | extracted SERVES | fwd | 1 |
| SERVES | same relation | fwd | 21 |
| SERVES | no triple | - | 6 |

  DEVELOPS by sentence template of the gold evidence quote (seed 7 (kb_dev)); <S>=gold source name or pronoun, <T>=gold target name
| template | fwd | REV | no triple |
|---|---|---|---|
| <S> is directly responsible for developing <T> | 0 | 6 | 0 |
| <S> builds and maintains <T> | 0 | 5 | 0 |
| <S> spearheads the technical development of <T> | 3 | 0 | 0 |
| <S> designs and engineers <T> | 0 | 1 | 0 |

**seed 11 (kb_seed11)**

L3 (seed 11 (kb_seed11)): extracted triples joining the two endpoints of a gold edge (in the supporting doc): direction vs gold
| gold relation | extracted relation | direction | count |
|---|---|---|---|
| BELONGS_TO | extracted MANAGES | fwd | 11 |
| BELONGS_TO | same relation | fwd | 7 |
| BELONGS_TO | extracted MANAGES | REV | 2 |
| BELONGS_TO | no triple | - | 2 |
| BELONGS_TO | extracted DEVELOPS | fwd | 1 |
| BELONGS_TO | extracted SERVES | fwd | 1 |
| BUILT_WITH | same relation | fwd | 6 |
| BUILT_WITH | same relation | REV | 3 |
| BUILT_WITH | no triple | - | 2 |
| BUILT_WITH | extracted DEVELOPS | fwd | 1 |
| DEVELOPS | same relation | REV | 11 |
| DEVELOPS | same relation | fwd | 4 |
| MANAGES | same relation | fwd | 11 |
| MANAGES | extracted SERVES | fwd | 4 |
| MANAGES | same relation | REV | 3 |
| MANAGES | extracted BELONGS_TO | fwd | 2 |
| MANAGES | extracted DEVELOPS | fwd | 2 |
| MANAGES | no triple | - | 1 |
| SERVES | same relation | fwd | 24 |
| SERVES | no triple | - | 3 |

  DEVELOPS by sentence template of the gold evidence quote (seed 11 (kb_seed11)); <S>=gold source name or pronoun, <T>=gold target name
| template | fwd | REV | no triple |
|---|---|---|---|
| <S> spearheads the technical development of <T> | 3 | 1 | 0 |
| <S> is directly responsible for developing <T> | 1 | 3 | 0 |
| <S> builds and maintains <T> | 0 | 4 | 0 |
| <S> designs and engineers <T> | 0 | 3 | 0 |

### What this does NOT show (Part 2)

- It does not show *why* the model reverses these sentences. Only parsed triples are stored, not the raw model output; the template association is a count over four templates and 15 edges per seed, with no test, and the "spearheads ..." sentences also differ in wording from the others.
- It does not show that the reversal rate is a property of the model for real text: the corpora are synthetic with one active-voice sentence per DEVELOPS edge; passive constructions ("X is developed by Y") do not occur.
- It does not show that repairing the direction improves retrieval; Table 3.2 and L4 report edge fate and path completeness (an upper bound), no retrieval was re-run on a repaired graph.
- It does not cover reversals that the ontology cannot see: where both orientations satisfy the rule (for example MANAGES EMPLOYEE -> EMPLOYEE) a reversed triple is validated and wrong; such triples are counted in Table 2.5 (same-relation REV rows for MANAGES) but cannot be detected from types.
- The seed-7 numbers come from the stored `triples.jsonl` of `kb_dev`; the original HEAD KB (`kb_before`) gives the same direction counts (see `part2_tables.txt`).

## Candidate fixes (ranked; none implemented)

### Table 3.1 Ontology-guided direction repair, computed on stored triples (EXPLORATORY, NOT A FIX, SEED ALREADY USED)

**seed 7 (kb_dev)**

```
L4 (seed 7 (kb_dev)) EXPLORATORY, NOT A FIX, SEED ALREADY USED: swap source/target of ontology-rejected triples if the swapped pair satisfies the same relation's rule
  triples with final status REJECTED_ONTOLOGY: 17; swap passes the rule: 10; of these, swapped triple equals a gold edge (same relation): 10; swapped triple matches no gold edge: 0
  | relation | mention types (as extracted) | swap allowed | swapped equals a gold edge | count |
  |---|---|---|---|---|
  | DEVELOPS | PRODUCT->DEPARTMENT | True | True | 7 |
  | DEVELOPS | EMPLOYEE->PRODUCT | False | False | 3 |
  | SERVES | EMPLOYEE->CLIENT | False | False | 2 |
  | MANAGES | DEPARTMENT->EMPLOYEE | True | True | 1 |
  | BELONGS_TO | DEPARTMENT->EMPLOYEE | True | True | 1 |
  | BUILT_WITH | TECHNOLOGY->PRODUCT | True | True | 1 |
  | MAINTAINES | DEPARTMENT->PRODUCT | False | False | 1 |
  | MAINTAINS | DEPARTMENT->PRODUCT | False | False | 1 |
  gold edges that would become exact-validated: 9 ['EDGE_0001', 'EDGE_0002', 'EDGE_0004', 'EDGE_0008', 'EDGE_0012', 'EDGE_0013', 'EDGE_0015', 'EDGE_0020', 'EDGE_0046'] (fates before: {'C': 8, 'B': 1})
  exact validated gold edges (A+E) before: 47; after swap (upper bound, ignores the D-fate edges and endpoint splits): 56
  | group | N | path-complete before | path-complete after swap |
  |---|---|---|---|
  | hop1 | 30 | 17 | 21 |
  | hop2 | 30 | 4 | 8 |
  | hop3 | 30 | 1 | 3 |
  questions that become path-complete: ['Q_0038', 'Q_0039', 'Q_0048', 'Q_0059', 'Q_0079', 'Q_0090']
  of those, currently succeeding in hybrid: 2/6; graph_only: 2/6
```

**seed 11 (kb_seed11)**

```
L4 (seed 11 (kb_seed11)) EXPLORATORY, NOT A FIX, SEED ALREADY USED: swap source/target of ontology-rejected triples if the swapped pair satisfies the same relation's rule
  triples with final status REJECTED_ONTOLOGY: 24; swap passes the rule: 15; of these, swapped triple equals a gold edge (same relation): 14; swapped triple matches no gold edge: 1
  | relation | mention types (as extracted) | swap allowed | swapped equals a gold edge | count |
  |---|---|---|---|---|
  | DEVELOPS | PRODUCT->DEPARTMENT | True | True | 11 |
  | SERVES | EMPLOYEE->CLIENT | False | False | 3 |
  | MANAGES | DEPARTMENT->EMPLOYEE | True | True | 2 |
  | DEVELOPS | EMPLOYEE->DEPARTMENT | False | False | 1 |
  | SERVES | EMPLOYEE->DEPARTMENT | False | False | 1 |
  | DEVELOPS | EMPLOYEE->PRODUCT | False | False | 1 |
  | BUILT_WITH | TECHNOLOGY->PRODUCT | True | True | 1 |
  | BUILT_WITH | PRODUCT->TECHNOLOGY | False | False | 1 |
  | DEVELOPS | PRODUCT->TECHNOLOGY | False | False | 1 |
  | BUILT_WITH | TECHNOLOGY->PRODUCT | True | False | 1 |
  | MAINTAINED | PRODUCT->DEPARTMENT | False | False | 1 |
    swap passes but is not a gold edge: D044_C0000_R001 BUILT_WITH TECHNOLOGY->PRODUCT gold endpoints None -> ENT_0044
  gold edges that would become exact-validated: 14 ['EDGE_0001', 'EDGE_0002', 'EDGE_0003', 'EDGE_0006', 'EDGE_0007', 'EDGE_0008', 'EDGE_0009', 'EDGE_0010', 'EDGE_0011', 'EDGE_0012', 'EDGE_0015', 'EDGE_0017', 'EDGE_0066', 'EDGE_0069'] (fates before: {'C': 14})
  exact validated gold edges (A+E) before: 42; after swap (upper bound, ignores the D-fate edges and endpoint splits): 56
  | group | N | path-complete before | path-complete after swap |
  |---|---|---|---|
  | hop1 | 30 | 12 | 16 |
  | hop2 | 30 | 2 | 12 |
  | hop3 | 30 | 0 | 5 |
  questions that become path-complete: ['Q_0032', 'Q_0034', 'Q_0035', 'Q_0038', 'Q_0042', 'Q_0043', 'Q_0046', 'Q_0054', 'Q_0056', 'Q_0060', 'Q_0069', 'Q_0075', 'Q_0081', 'Q_0082', 'Q_0090']
  of those, currently succeeding in hybrid: 5/15; graph_only: 6/15
```

**1. Direction repair at validation, behind a config flag (default off).** When a triple is rejected for ontology reasons and swapping source and target satisfies the same relation's allowed source and target types, swap and revalidate (the evidence-quote checks stay as they are); log every swap.

- Expected effect (computed above, not measured in retrieval): of 24 ontology-rejected triples on seed 11, 15 pass after the swap, 14 of them equal a gold edge, which would move 14 gold edges to exact validated (42 -> 56); on seed 7, 10 of 17 pass and 10 equal a gold edge (9 gold edges to exact, 47 -> 56). Questions with a complete exact gold path (upper bound for graph reach): hop2 2 -> 12, hop3 0 -> 5 (seed 11); hop2 4 -> 8, hop3 1 -> 3 (seed 7).
- Edges it would change: seed 11 ['EDGE_0001', 'EDGE_0002', 'EDGE_0003', 'EDGE_0006', 'EDGE_0007', 'EDGE_0008', 'EDGE_0009', 'EDGE_0010', 'EDGE_0011', 'EDGE_0012', 'EDGE_0015', 'EDGE_0017', 'EDGE_0066', 'EDGE_0069']; seed 7 ['EDGE_0001', 'EDGE_0002', 'EDGE_0004', 'EDGE_0008', 'EDGE_0012', 'EDGE_0013', 'EDGE_0015', 'EDGE_0020', 'EDGE_0046']. Questions that become path-complete: seed 11 ['Q_0032', 'Q_0034', 'Q_0035', 'Q_0038', 'Q_0042', 'Q_0043', 'Q_0046', 'Q_0054', 'Q_0056', 'Q_0060', 'Q_0069', 'Q_0075', 'Q_0081', 'Q_0082', 'Q_0090'] (hybrid currently succeeds on 5 of 15); seed 7 ['Q_0038', 'Q_0039', 'Q_0048', 'Q_0059', 'Q_0079', 'Q_0090'] (hybrid currently succeeds on 2 of 6). Path completeness is not recall: the graph route also needs the seed and the ranking to cooperate.
- Risk: 1 of 15 swaps on seed 11 and 0 of 10 on seed 7 produce an edge that is not a gold edge (a BUILT_WITH triple whose source mention maps to no gold entity); it does not fire when both orientations are valid; it assumes the stated relation is right and only the direction is wrong; the swap cannot be verified from the quote text. Needs a unit test for the flag-off identity and a domain-agnostic implementation (ontology rules only).

**2. Route-weighted RRF (config key such as `rrf_graph_weight`, default 1.0 = current behaviour).**

- Expected effect (Table 1.5, K=4): hop3 1 -> 4 (seed 7), 0 -> 1 (seed 11); hop2 18 -> 19 and 11 -> 13; no baseline-correct question lost on either seed. Questions changed: hop3 ['Q_0066', 'Q_0067', 'Q_0089'] (seed 7), ['Q_0073'] (seed 11); hop2 ['Q_0045'] (seed 7), ['Q_0034', 'Q_0036'] (seed 11).
- Risk: the weight is chosen after seeing the data (weights 2-5 give identical results, so the choice is not sensitive on these questions, but that is two seeds of one synthetic benchmark); a graph-biased ranking can displace correct vector chunks on corpora where the graph route is noisy; the unanswerable group and answer quality were not evaluated; the gain is small compared with the candidate-generation losses of Part 1 (mechanism A), so it cannot lift hop3 far above the ceiling set by the union of the two lists.

**3. Ontology-derived direction instruction in the relationship prompt (new versioned prompt file, selected by `relationship_extraction_version`).** Generate, from the ontology, an explicit sentence per relation (source types, target types, "the subject of an active sentence is the source") and add a worked example built from the ontology's own description.

- Expected effect: should address the same 23 reversed DEVELOPS triples (and the 5 + 5 + 1 reversed MANAGES, BUILT_WITH and BELONGS_TO triples) at the source, but the prompt already contains two direction statements that the model did not follow, and the earlier prompt change (v1.1) failed its dev rule; the expected effect is therefore uncertain and unmeasured.
- Risk: requires re-extraction (about 45 minutes per 60-document corpus on this machine), changes extraction for relations that are currently correct, and overlaps with fix 1; it should be considered only for the residual after fix 1 and must be pre-registered like any prompt change.

Ranking rationale: fix 1 changes the largest number of edges (the only cause found for 11 of 15 DEVELOPS rejections), needs no re-extraction (it works on the stored triples) and its flag-off path is identical to current behaviour; fix 2 is cheap and its measured effect is small and positive but bounded by candidate generation; fix 3 is the most expensive and the least certain. Per the protocol note, all three would be confirmed on seed 13 only, pre-registered before the fix is applied.

## Files

`part1_record.py` (recording), `part1_records_seed{7,7mo,11}.jsonl` (per question and mode: graph list, vector list, merged ranking, context), `part1_analyze.py` → `part1_tables.txt`, `part1_summary.json`, `part1_hop3_detail_seed*.txt`; `part1c_traversal.py` → `part1c_tables.txt`, `part1c_summary.json`; `part2_develops.py` → `part2_tables.txt`, `part2_edges.json`; `link_analysis.py` → `link_tables.txt`, `link_summary.json`; `build_diagnosis.py` → this file; `hash_before.txt`, `hash_after.txt`.

