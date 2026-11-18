# Hybrid GraphRAG engine: project context

## Goal
A complete, modular, domain-agnostic Hybrid GraphRAG engine, evaluated with scientific integrity.
Pipeline: load -> chunk -> entity extraction -> relationship extraction -> entity resolution -> graph build/validation
(python-igraph) -> embeddings (ChromaDB) -> Leiden communities + reports.
Query: plan -> seed resolution -> graph route / vector route / community route -> merge by chunk ID -> RRF (k=60)
-> context -> cited answer or abstention.
Public API: hybrid_graphrag.HybridGraphRAG (from_config, from_existing, ingest, query(mode: hybrid |
vector_only | graph_only | community_aware), get_kb_stats, close).
Stack: gemma3:4b and embeddinggemma (768d) via Ollama. MacBook Air M2 16GB, no GPU. Extraction is slow
(about 45 min per 60 short docs), so prefer retrieval-only evaluation and avoid unnecessary re-ingestion.

## Repo state
Branch hybrid-graphrag-build, HEAD 3877a4d. Nothing pushed.
Untracked: experiments/hotpotqa_m3/ (contains a 51 MB data/ dir; never commit it), experiments/hotpotqa_m4/,
experiments/synth_kg/, experiments/synth_kg_audit/.
Baseline tests: 194 passed, 1 skipped.
Configs: configs/default.yaml, configs/domains/{academic,company}.yaml.
Prompts are versioned in prompts/. The answer prompt v1.1 is used in experiments; v1.0 is the default.

## Rules for working here
- Do not commit, push, stash, reset or clean unless the user asks. Ask before committing.
- Never trust "tests pass" or a narrative summary alone. Verify numbers from scripts that read files on disk.
- Evaluation hygiene: ERROR counted separately from abstentions; T=0; equal limits across modes; Wilson CIs;
  exact sign tests on paired discordant questions; Holm correction across families of tests; decision rules written
  before the run; no tuning after seeing results; any fix found on one corpus is confirmed once on a fresh seed.
- Scripts and protocol files that produced reported numbers must not be edited after the runs without a failing
  test and a note in the decision log.
- The engine must stay domain-agnostic: no hardcoded entity-type literals (e.g. "PERSON") in src/.
- Keep defaults backward compatible; new behaviour goes behind config flags.
- The user wants explanations of the engineering, one recommendation with reasoning, and prompts or plans that are
  self-contained. Technical cause-and-effect explanations, no analogies.

## What has been established (evidence in experiments/)
- Generator ceiling: with gold context, gemma3:4b reaches about 80% on HotpotQA (strict "contains" metric understates).
- HotpotQA is a poor benchmark for this setup (63.5 s/paragraph ingestion, about 82% relationship rejection).
  The project moved to a procedural synthetic benchmark: experiments/synth_kg/ (seed 7, company.yaml ontology,
  60 docs, 120 questions, gold KG).
- Seed-7 results: hop2 accuracy vector 6/30, hybrid 18/30, graph_only 16/30, community_aware 17/30; hop3 about 2/30
  in all modes. Hybrid beats vector on hop2 after Holm correction (adjusted p = 0.0275).
- BUT graph-free baseline B9 (dense top-1 seed, capitalized-name overlap expansion, 3 candidates) reached 21/30
  hop2 evidence recall against hybrid's 17/30, so by the pre-registered rule "graph-specific benefit not
  demonstrated" on this corpus (experiments/synth_kg_audit/followup/output.txt, Part F).
  Caveat: B8 vs B9 differ only in seed/candidate counts and score 8/30 vs 21/30; B4 did not reproduce (9/30 vs 11/30).
- Verified engine defects:
  1. Hardcoded "PERSON" in src/resolution/entity_resolver.py (lines ~208-215) and src/retrieval/seed_retriever.py
     (lines ~136-138); company ontology uses EMPLOYEE, so surname/initial matching never runs.
  2. require_type_match (entity_resolver.py:~327) splits entities whose mentions were typed inconsistently
     (15 gold entities split).
  3. Relationship ontology validation runs per chunk on mention types, before resolution
     (relation_extractor.py:~332-339). Employee names typed PRODUCT turn MANAGES into DEVELOPS.
  4. In 9 docs the text states two relations for one entity pair; only one was extracted.
  5. No coreference step ("They is an official organizational part of ..." loses the edge).
  6. Hop3 fails because of extraction losses and split-entity sink vertices, not context truncation.
  7. "[GRAPH_EDGE_PROVENANCE: ...]" marker text leaks into answers; wrong KG edges reach answers.
- Benchmark issues: 14 of 15 alias strings appear in no document; Q_0091 and Q_0094 gold entity names may not
  match corpus names; 28 of 32 "managed_client" unanswerable evaluations are near-answerable (corpus says
  "serves"); the "validated path exists" flag in the original report is inconsistent with the later audit.
- Reasoning about next step: after extraction fixes, test on a fresh seed whether hybrid beats B9 (hop2/alias)
  and a two-step baseline B10 (hop3), with pre-registered rules.

## Current task
Implement and evaluate the fixes for defects 1-4 (Fix 1: person_like_types from ontology; Fix 2: type
reconciliation + relationship re-validation after resolution, behind flags; Fix 3: multi-relation prompt, only if the
current prompt forbids it), measure on seed 7 without LLM calls, then confirm once on seed 11 (retrieval-only,
protocol written before generation, frozen diff hash). The full spec is in docs/NEXT_TASK.md (paste it there).
Also open: commit the experiment directories (needs user approval), M5 reusability (open-domain ontology
config), M6 docs and release.

## Status (2026-11-16, after the synth_kg_fix task; evidence in experiments/synth_kg_fix/FINAL_REPORT.md)
- Implemented, behind config flags (defaults unchanged): person-like types from the ontology (`person_like_types`), initial+surname matching, merge of identical names across types with type vote, post-resolution relationship re-validation, prompt-version key (v1.1 file exists, not enabled), `graph_context_mode: "off"` omits the graph block. Suite: 224 passed, 0 skipped.
- Seed 7 (dev, retrieval-only, K=4): hybrid alias 40.0% -> 66.7%, hop2 56.7% -> 60.0%, hop3 3.3% -> 3.3%; graph_only alias 0% -> 60%, hop3 3.3% -> 20%. Gold entities split 15 -> 0. Revalidation removed 3 wrong-type edges but cut graph_only hop3 (33% without it, 20% with it).
- Seed 11 (fresh, pre-registered): hybrid hop2 36.7%, hop3 0.0%, alias 80.0%. B9 hop2 76.7% (Holm p = 0.0015 in B9's favour), B10 hop3 16.7%, B9 alias 73.3%. Decisions: P1 and P2 "graph-free baseline matches or exceeds hybrid", P3 "inconclusive". Graph-specific benefit is NOT demonstrated on this benchmark.
- Prompt v1.1 failed its dev rule (gold edges recovered +1, needed +3) and stays disabled.
- Open: why hybrid hop3 is 0-1/30 although graph_only reaches 6-10/30 on seed 7 (graph evidence does not reach the 4 context chunks); 11 of 15 gold DEVELOPS edges rejected by the ontology on seed 11; TECHNOLOGY mistyped as PRODUCT after type vote (3 entities on seed 11); partial-name aliases ("X Division", product stems); no coreference; revalidation vs traversal-connectivity trade-off; commit of experiment directories (needs user approval); M5 reusability; M6 docs and release.
