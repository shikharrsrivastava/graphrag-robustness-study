# Pre-registration: fresh-seed confirmation (seed 11)

Written before the seed-11 corpus was generated. Nothing below is changed after generation.

## Corpus
- Generator: `experiments/synth_kg/generate.py` with `--seed 11` and exactly the knobs used for `corpus_final`
  (`experiments/synth_kg/corpus_final/GENERATED_BY.txt`; `experiments/synth_kg/DECISIONS.md`, entry 2026-11-16T03:03:46):
  `python3 experiments/synth_kg/generate.py --seed 11 --n_docs 60 --alias_rate 0.3 --pronoun_rate 0.2 --distractor_rate 0.3 --out experiments/synth_kg_fix/corpus_seed11`
- Then `python3 experiments/synth_kg/validate.py --corpus-dir experiments/synth_kg_fix/corpus_seed11`.
- No calibration, no re-rolling. If validation fails, the task stops.

## Engine
- Public API (`hybrid_graphrag.HybridGraphRAG.from_config` + `ingest`), `configs/default.yaml` + `configs/domains/company.yaml`,
  gemma3:4b, embeddinggemma, T=0, with the same run overrides as `experiments/synth_kg/run_overnight.py:97-114`
  (`vector_top_k=4`, `final_context_chunks=4`). Script: `experiments/synth_kg_fix/ingest_seed11.py`. Single ingestion attempt.
- Fixed code. The frozen diff hash (`git diff -- src tests configs prompts | shasum -a 256`) is recorded in
  `experiments/synth_kg_fix/frozen_diff_sha.txt` before generation and recomputed after the final evaluation.
- The fixes are tested as a bundle: the Fix 1 extension (person-like types from the ontology, including the relation
  extractor), the Fix 2 flags (`merge_exact_name_across_types`, `revalidate_relationships_after_resolution`,
  `enable_initial_surname_person_merge`) and, if enabled by the Phase 2c rule, the Fix 3 prompt all apply during
  seed-11 extraction and resolution. No per-fix attribution is possible on seed 11.

## Methods (K = 4 documents, answerable questions only)
- **hybrid** (retrieval only) and **vector_only**: engine query path through the public API with the query planner's
  plans cached on disk and answer generation replaced by a stub (`retrieval_eval.py`). Retrieved documents = documents
  of the returned context chunks (`final_context_chunks=4`).
- **B1 dense top-4**: the 4 documents with the highest cosine similarity to the question (embeddinggemma, full document text).
- **B7 iterative dense**: d1 = dense top-1; q2 = question + " " + text(d1) (first 2,000 characters); return d1 plus the
  top 3 other documents by cosine to q2.
- **B9**: dense top-1 seed; extract capitalized-name runs from the seed document (regex
  `\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b`), excluding names that appear in the question (case-insensitive
  substring); candidate documents = other documents containing any remaining name, ranked by the number of distinct
  remaining names, ties by cosine to the question; return the seed plus the top 3. Fewer than 3 candidates: fill from
  the dense ranking.
- **B10 two-step overlap**: seed = dense top-1; step 1 as B9 but keep only the top-1 candidate; step 2 repeats the
  name-overlap expansion from that candidate, excluding names in the question and in the seed, and adds the top 2
  candidates; total 4 documents. Fewer than 4 documents: fill from the dense ranking.
- Implementation: `experiments/synth_kg_fix/fixlib.py` (B7 and B9 copied from
  `experiments/synth_kg_audit/followup/part_f_iterative_baselines.py`), evaluation `experiments/synth_kg_fix/seed11_eval.py`.

## Metric
All-gold evidence recall: a question succeeds when every `gold_supporting_doc_ids` document is among the returned
documents. Reported per group (hop1, hop2, hop3, alias) with Wilson 95% intervals. A query ERROR is counted as not
retrieved and reported separately. T=0 everywhere.

## Primary outcomes
- **P1**: hybrid vs B9 on hop2.
- **P2**: hybrid vs B10 on hop3.
- **P3**: hybrid vs B9 on alias.

Each is tested with a paired exact two-sided sign test on per-question success (discordant pairs), Holm-corrected over
the 3 tests at alpha 0.05.

## Decision statements (per primary outcome)
- "graph advantage demonstrated" if hybrid minus the baseline is at least 10 points AND the Holm-adjusted p < 0.05;
- "graph-free baseline matches or exceeds hybrid" if the baseline is at least hybrid minus 5 points;
- otherwise "inconclusive".

## Descriptive only (no decision)
- hybrid vs vector_only per group.
- The Phase 2b-style tables on seed 11: gold-edge fate, entity splits, alias seeds (`kb_tables.py`).

## Not allowed
- No change under `src/`, `tests/`, `configs/`, `prompts/` after the frozen hash is recorded.
- No re-run of seed 11, no other seed, no tuning after seeing results.
- If ingestion fails or is interrupted: report and stop.
