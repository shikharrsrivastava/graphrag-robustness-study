# CHECKPOINT 0 - Safety, provenance and Phase 0 re-verification

## Part A
- A1: no Antigravity / agy / gemini process running. Snapshot `git diff | shasum -a 256` = `c511f995...87ff18` at 18:44; the 6 pre-existing edits kept their 14:20-14:24 mtimes until my first edit.
- A2: `~/graphrag_backup_20261116_e` equals HEAD for every tracked file under src, tests, configs, prompts (cmp, 0 differences). Not overwritten. New backup `~/graphrag_backup_20261116_f` taken before my first edit.
- A3: pre-existing uncommitted edits saved to `preexisting_edits.diff` (+ `preexisting_relationship_extraction_v1_1.txt`), reviewed hunk by hunk and adopted. Removed: duplicate top-level `EngineConfig.revalidate_relationships_after_resolution`. Changed: `company.yaml` no longer sets `relationship_extraction_version: v1.1` (decided by the Phase 2c rule).
- A4: the test that moved from skipped to passing is `tests/test_embeddings_and_resolution.py:553` (`skipif(not is_live_ollama_available())`). Ollama is running now, so the suite is 195 passed, 0 skipped at the start of this task.

## Part B (full text: `step0_reverify.txt`; data: `step0_reverify_data.txt`, `step0_b7_data.txt`; script: `step0_reverify.py`)
Prior claims: 14 CONFIRMED, 6 IMPRECISE, 3 REFUTED (table at the top of `step0_reverify.txt`).

Key numbers (all recomputed from disk):
- 98 raw triples: 55 VALIDATED, 15 REJECTED_ONTOLOGY, 25 REJECTED_ENDPOINT_NOT_IN_EVIDENCE, 3 REJECTED_QUOTE_MISMATCH. 3 unordered mention pairs carry two relation types, so prompt v1.0 does not prevent several relations per pair.
- 14 of 15 alias strings occur in no document. "Galthor Nebikvane" / "Lorik Sarkithtex" come from `synth_kg_audit/followup/part_c_alias_audit.py:29,32` and exist nowhere else; the gold names match the corpus.
- B4: 9/30 vs 11/30 reproduced; the cause is question-name removal; the affected questions are Q_0031 and Q_0052 (prior text said Q_0044, Q_0055).
- "validated path exists": undirected untyped BFS (`score.py:350-466`); the four flagged questions are joined through shared CLIENT/DEPARTMENT hubs, with no gold-path edge on the path.
- Provenance marker in answers: hybrid 7, graph_only 8, community_aware 2, vector_only 0. Cause: `graph_context_mode=False` takes the legacy path block in `context_builder.py`.
- 15 gold entities split (6 EMPLOYEE/PRODUCT, 6 TECHNOLOGY/PRODUCT, 3 other pairs). 4 (not 5) EMPLOYEE->EMPLOYEE MANAGES edges stored as DEVELOPS, plus 1 as SERVES (rejected). 10 (not 9) documents state two relations for one pair.
- hybrid hop2 17/30, B9 hop2 21/30 confirmed.
- No config-hash check in `from_existing`. No retrieval-only hybrid entry point existed; `retrieval_eval.py` was written. The run records contain no query plans; 105 plans were generated and cached (`plans_seed7.json`, 0 errors, all LOCAL). Replay of HEAD code with cached plans reproduces the recorded context documents on 105/105 questions in hybrid, graph_only and vector_only.

## Skipped / failed
- Nothing skipped. Phase 0 took longer than the 20-minute budget because the plan cache (105 planner calls) shared Ollama with a control rebuild.
