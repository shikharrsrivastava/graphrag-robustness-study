## 6. Skipped, failed, or deviating from the prompt

1. **Unlogged pre-existing edits** (14:20-14:24, prior agent) were reviewed and adopted (`preexisting_edits.diff`). Duplicate top-level `revalidate_relationships_after_resolution` removed; `relationship_extraction_version: v1.1` removed from `company.yaml` until the dev rule decided (it failed, so v1.0 stays).
2. **Extra flag added** (not in the prompt): `resolution.enable_initial_surname_person_merge` (default false, true in company.yaml). It controls only mention merging in resolution; it produced 0 merges on seed 7 and the seed-11 report lists none, because no document contains an initial+surname form. Seed retrieval does not depend on it.
3. **Behaviour change without a flag:** `person_names_compatible` accepts initial+surname for every ontology (as specified); for PERSON ontologies this changes seed retrieval ("J. Smith" now matches "John Smith").
4. **Fix 4 scope:** only the string `"off"` omits the graph block; boolean `False` keeps the legacy block because an existing test pins it and it is the default. The seed-7 overnight runs used boolean `False`.
5. **Phase 0 verdicts** are written to `step0_reverify.txt`, not into `step0.txt` (the task said not to delete or overwrite the prior agent's files). Phase 0 exceeded the 20-minute budget because a 105-call planner cache shared Ollama with a control rebuild.
6. **Query plans**: the seed-7 run records contain no plans, so plans were generated once by the planner (T=0) and cached; replay with HEAD code reproduces the recorded context documents on 105/105 questions in hybrid, graph_only and vector_only. Seed-11 plans were generated once after ingestion.
7. **No retrieval-only entry point existed** in `synth_kg_audit/followup/`; `retrieval_eval.py` was written (engine query path unchanged; planner and answer generator replaced; LLM client raises on any call).
8. **"Before" numbers** come from HEAD code (`head_tree/`, copied from the verified backup) running on a copy of the original KB (`kb_before/`, snapshot storage_root repointed). The original `from_existing` keeps an absolute storage_root, so copying a KB silently reopens the original unless the snapshot is edited.
9. **Phase 2c document set**: T1 = 10 documents, T2 = 10 documents (14 edges), union 15, plus 6 controls = 21 documents (not "9 + 14 + 6").
10. **Hash**: the specified `git diff | shasum` does not see untracked files; a supplementary all-files hash was recorded too. Both are unchanged.
11. **Validator invocation**: `validate.py` needs `PYTHONPATH` set to the repository root; the first call failed on an import, the corpus itself passed.
12. **Sleep interruption** (~19:26): no job was killed or lost (both rebuilds had already completed); nothing was re-run. Logged in `PROGRESS.md`.
13. **Extra, non-pre-registered variant**: `kb_dev_merge_only` (revalidation off) and `B10` on seed 7 were computed for information only; neither feeds a decision.
14. **B7 second-stage query**: the stored seed-7 vectors are reproduced exactly only when the raw (unstripped) document text is appended; `embed_corpus.py` uses that and was verified against the stored vectors (cosine >= 0.9999999) before use on seed 11.
15. **`seed11_eval.py`** was edited once before its first run to read the combined results file `retrieval_seed11_all.jsonl` (hybrid, graph_only, vector_only in one run).
16. **Not investigated** (outside the task): why hybrid hop3 is 0-1/30 when graph_only reaches 6-10/30 on seed 7; why 11 of 15 DEVELOPS gold edges are rejected by the ontology on seed 11; the `"PROJECT"` literal in `entity_resolver.py`.
17. Claims taken from CLAUDE.md and not re-verified here: "28 of 32 managed_client unanswerable evaluations are near-answerable", the 0.0275 Holm-adjusted p for hybrid vs vector on seed-7 hop2.

## 7. Limitations (what the numbers show)

- **One ontology** (company.yaml, five entity types), **one dev seed and one fresh seed**, N = 30 per group (15 for alias): Wilson intervals are wide (hop2 hybrid on seed 11: 21.9-54.5%).
- **Names are procedurally unique** (no homonyms across entities), which favours name-overlap baselines and exact-match resolution. B9 and B10 exploit this directly.
- **Flags chosen on dev data**: the enabling of the Fix 2 flags was decided on seed 7; seed 11 tests the bundle, so no per-fix attribution is possible.
- **Seed 11 vs seed 7 for the same engine**: hybrid hop2 56.7% (seed 7, before fixes) / 60.0% (seed 7, after) vs 36.7% (seed 11); hop3 0-3.3% on both; alias 66.7% vs 80.0%. The seed-to-seed spread is of the same size as the effects being tested.
- **Pre-registered outcome**: P1 and P2 baseline matches or exceeds hybrid; P3 inconclusive. The data do not support a graph-specific advantage over graph-free baselines on this benchmark. The hybrid route beats plain dense retrieval on hop2 (7 vs 0 discordant questions, unadjusted p = 0.0156) and on alias (5 vs 0, p = 0.0625), descriptive only.
- **Retrieval metric only**: document-level all-gold recall at K = 4; answer quality, citation behaviour and the provenance-marker fix were not evaluated (no generation in this task).
- **Revalidation trade-off** (seed 7): +1 exact gold edge and -3 wrong-type edges, but -3 traversal links; graph_only hop3 20% with it vs 33% without. Seed 11 shows the same mechanism on 4 changed triples (one a consequence of mistyped TECHNOLOGY entities).
- **Type vote weakness**: ontology declaration order breaks ties between PRODUCT and TECHNOLOGY mentions towards PRODUCT (3 gold TECHNOLOGY entities on seed 11), because PRODUCT is declared earlier.
- Prompt v1.1 was evaluated on 21 seed-7 documents only; it was not run on seed 11.
