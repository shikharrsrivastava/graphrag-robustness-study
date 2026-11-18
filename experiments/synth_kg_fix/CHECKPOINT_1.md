# CHECKPOINT 1 - Fixes

Test suite: **224 passed, 0 skipped, 0 failed** (195 existing + 29 new in `tests/test_synth_kg_fixes.py`). No existing test was edited.
The new test file fails at import on HEAD code (checked against `head_tree/`), so it exercises the new code paths.

## Fix 1 - person-like types from the ontology
- `src/core/ontology.py:28` `DEFAULT_PERSON_LIKE_TYPES`; `:50` field `person_like_types`; `:69-80` validator (listed types must exist unless the default is in use); `:82` `is_person_like_type`; `:87` `entity_type_rank`.
- `src/resolution/entity_resolver.py`: every `"PERSON"` comparison replaced by `self._is_person(...)` (`:199`; uses at `:257, :262, :602, :610, :639, :643, :667, :717, :744, :776-777`).
- `src/resolution/entity_resolver.py:85-143`: `_person_name_tokens`, `is_initial_surname_form`, initial+surname rule inside `person_names_compatible` (`:132-137`).
- Resolution merge of initial+surname mentions, only with exactly one candidate: `_unique_initial_surname_target` (`:307`), sequential pass (`:621-628`), post-pass (`:285-303`). Behind new flag `resolution.enable_initial_surname_person_merge` (default False; `src/core/configuration.py:83`), because merging in resolution is new behaviour for every ontology.
- `src/retrieval/seed_retriever.py:48` new constructor argument `person_like_types`; `:79` `_is_person`; `:147-149` and `:305` use it. The step-2 branch (`:305`) runs for every person-like entity whatever `expected_type` is (None or wrong type included), so alias seeds without a planner type still match. The step-1 gate (`:147`) is unchanged in structure so PERSON ontologies behave as before.
- `src/engine.py:351` passes `ontology.person_like_types` to the seed retriever.
- `src/extraction/relation_extractor.py:198, :221, :277` use `self.ontology.is_person_like_type`.
- Left alone as instructed: literal `"PROJECT"` at `src/resolution/entity_resolver.py:783` (fallback descriptors for PROJECT). It is still a hardcoded entity-type literal in src/.
- Tie handling in seed retrieval (unchanged policy): candidates sorted by entity_id, capped at `max_candidates_per_seed`, flagged AMBIGUOUS when more than one matches; all capped candidates are passed to traversal.
- Behaviour change NOT behind a flag (by instruction): `person_names_compatible` now accepts initial+surname, so seed retrieval for PERSON ontologies also resolves "J. Smith" to "John Smith".

## Fix 2 - type reconciliation and re-validation
- `merge_exact_name_across_types` (default False): `entity_resolver.py:585-589` (exact normalized name merges across types), `:249-251` + `_reconcile_entity_types` (`:331-430`): plurality vote; tie -> (1) type keeping most VALIDATED/REJECTED_ONTOLOGY relationships valid, (2) ontology declaration order; tied neighbours treated as their first tied type in declaration order; votes stored in `CanonicalEntity.type_vote_counts` (`src/core/models.py:173`).
- `revalidate_relationships_after_resolution` (default False): `_revalidate_with_canonical_types` (`entity_resolver.py:530-580`), called in `resolve_and_remap` (`:440-452`) before the VALIDATED filter (`:457`). Only VALIDATED / REJECTED_ONTOLOGY triples are recomputed; original status kept in `Relationship.mention_level_status` (`src/core/models.py:202`).
- Persistence: `src/engine.py:178` + `_persist_resolution_outputs` (`:303-316`): re-validated triples replace `triples.jsonl`; `data/processed/resolution_report.json` holds votes, tie-breaks and status changes. Nothing extra is written with default flags.

## Fix 3 - prompt selection (not yet enabled)
- `src/engine.py:124` + `_relationship_prompt_kwargs` (`:278-301`): `prompt_versions.relationship_extraction_version` v1.0 -> historical default file, v1.1 -> `prompts/relationship_extraction_v1_1.txt`; a missing file raises. Validation of the key: `src/core/configuration.py:206-214`.
- `diff prompts/relationship_extraction.txt prompts/relationship_extraction_v1_1.txt`: one added instruction (10) and a blank line; nothing removed (also asserted by a test).
- Premise from Phase 0: prompt v1.0 is silent on several relations per pair, with instruction 8 steering to one type; the model emitted two types for one pair in 3 of 10 two-relation documents.
- Naming hazard: the older metadata label `prompt_versions.relationship_extraction` is "v1.1" for the file that the new key calls "v1.0".

## Fix 4 - provenance marker
- `src/retrieval/context_builder.py:38, :63`: the STRING "off" now omits the graph path block (community summaries kept). Boolean False keeps the legacy block, because `tests/test_milestone2_fixes.py:553-555` pins it and it is the default. The seed-7 runs used boolean False, so they would still leak; use `graph_context_mode: "off"` (or "none") in future generation runs. Not exercised in this task (no answer generation).

## Config
- `configs/default.yaml`: three new resolution keys, all false.
- `configs/domains/company.yaml`: `person_like_types: ["EMPLOYEE"]`, the three resolution flags true. Prompt version left at default v1.0 until Phase 2c.

## Skipped / failed
- Nothing skipped.

---
# Addendum (written after Phase 2b): is 53 relationships against 55 a net loss?

Source: `phase2b_tables.txt` (script `kb_tables.py`), `kb_dev/.../data/processed/resolution_report.json`.

**Answer: no loss of correct edges; a gain in precision and a loss of traversal connectivity.**

| quantity | before (55 edges) | merge + initial only (55) | all fixes = kb_dev (53) |
|---|---|---|---|
| gold edges validated with exact gold type (fate A+E) | 46 | 46 | **47** |
| validated edges matching no gold edge exactly | 9 | 9 | **6** |
| gold edges validated under another type, same endpoints (fate B) | 17 | 17 | 15 |
| gold edges rejected by ontology (fate C) | 11 | 11 | 12 |
| score.py relaxed recall (gold edges with any validated edge between the endpoints) | 54 | 54 | 52 |

Fate transitions before -> after (only changes): E->A 19 (split-endpoint edges now attach to one entity), C->A 1 (`EDGE_0062` Nebikvane MANAGES Zantonnax), C->B 1 (`EDGE_0041`), B->C 3 (`EDGE_0071`, `EDGE_0072`, `EDGE_0076`).

- The 55 -> 53 difference is: -3 validated edges, +1 validated edge, and a net of -1 elsewhere through merging.
  The 3 removed edges are `employee -DEVELOPS-> employee` triples (mention typed PRODUCT, canonical type EMPLOYEE after the vote; `resolution_report.json` revalidation changes D011_C0000_R002, D012_C0000_R002, D016_C0000_R002). They were wrong-type stand-ins for gold MANAGES edges. The revalidation also rescued 1 triple (D002_C0000_R001, MANAGES).
- They are removed by the ontology rule working as intended, and no gold edge with the exact gold type was lost. But they were the only validated link between those gold endpoints, so the graph lost 3 traversal links.
- Retrieval consequence on seed 7 (graph_only, vs the merge-only variant that keeps those edges): hop2 20/30 -> 18/30, hop3 10/30 -> 6/30; hybrid hop2 16/30 -> 18/30 (noise: 2 questions). Revalidation therefore costs graph_only recall on this corpus although it improves edge correctness. This is reported as a finding; the flag choice in `company.yaml` is left as specified (both flags on) because no decision rule for revalidation was written before the run, and changing it now would be selection on dev data.
- Type vote report: 15 entities had mixed mention types; 8 decided by plurality, 4 by relationship validity, 3 by ontology declaration order. One gold entity ends with the wrong type: Durnenlok Noxornok (gold EMPLOYEE, 1 EMPLOYEE vs 1 PRODUCT mention; tie broken by relationship validity towards PRODUCT because the one adjacent edge is MANAGES->PRODUCT-valid).
- `initial_surname_merges` in the resolution report is empty: no document contains an initial+surname form, so the alias gains come from seed retrieval, not from resolution.
