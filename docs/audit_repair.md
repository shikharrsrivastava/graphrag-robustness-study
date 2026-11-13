# Hybrid GraphRAG audit and repair

## Findings from the repository

- `index_directory` loaded files and chunks without assigning any `scope_id`; `query(scope_id=...)` therefore filtered Chroma records against metadata that did not exist. The document store had KB isolation but not question scope isolation.
- The original Hotpot runner shared one KB across questions. `PDFLoader` restarted `D001` numbering for every directory and the chunker generated IDs from that document ID, so repeated indexing replaced JSONL state and upserted collisions in Chroma. The runner's `gold title OR first two` condition admitted distractors.
- The academic ontology did not define `ACTED_IN`; the planner converted an all-invalid path into `requires_clarification`, which returned before vector retrieval.
- Hybrid retrieval called graph, vector, and community routes sequentially without exception isolation. Community summaries were global to a KB and could bring evidence chunks from another scope.
- Retrieval abstention was based on an empty list, although evidence presence does not prove answerability. Generation returned structured status, but the engine returned only answer text and did not synchronize that status into `RetrievalResult`.
- Reindexing overwrote JSONL files but left stale Chroma IDs behind. Per-chunk extraction errors also stopped the complete indexing run.

## Corrected data flow and contracts

| Stage | Input → output | IDs and metadata | Persistence / empty result |
|---|---|---|---|
| Loader | PDF/TXT/MD → `Document`, `Page` | KB ID, document ID, filename, source path, hashes, page number | JSONL; empty/scanned pages are recorded and expected |
| Chunker | Pages → `Chunk` | KB ID, document ID, chunk ID, page numbers, content hash | JSONL; empty pages yield no chunks |
| Extraction | Chunk → `EntityMention`, raw `Relationship` | Mention and relationship IDs derive from chunk ID; source chunk/doc/page provenance | JSONL; no entities/relations may be legitimate; failed chunks are reported and skipped |
| Resolution | Mentions + triples → canonical entities, mapping, remapped relationships | Canonical entity ID, aliases, source mention/chunk/doc IDs | Graph JSONL; unresolved mentions stay out of canonical graph |
| Embeddings | Canonical entities and chunks → vectors | Chroma IDs are entity/chunk IDs; metadata includes KB and chunk scope | Persistent Chroma upsert; no entities can be valid, but chunks should exist for non-empty input |
| Graph build/validation | Canonical entities + validated relationships → directed igraph + manifest | Nodes retain source chunk/doc IDs; edges retain relationship ID, source chunks/docs, relation type, scope | Graph records/manifest; zero edges means extraction/validation or source content must be investigated |
| Leiden/summaries | Graph → communities/reports | Partition and report IDs; supporting chunk IDs | JSONL + Chroma; empty graph can yield no communities |
| Query planning | Question → `QueryPlan` | Seed names/types, target type, ontology-valid relationship path; invalid hints recorded | LLM; planner failure now falls back to a safe mixed plan in hybrid modes |
| Seed resolution/traversal | Plan + canonical entities/graph → paths | Entity IDs and edge IDs; path scope filters edges | Graph only runs with resolved seed IDs; no paths is a legitimate route miss |
| Vector/community retrieval | Question embedding → evidence candidates | KB and optional scope metadata; evidence resolves to stored original chunks | Chroma; each route can fail independently and reports its error |
| Merge/RRF/context | Route evidence → deduplicated ranked evidence and context | Chunk ID is merge key; route/rank/path/citation provenance retained | In-memory; empty evidence is a retrieval miss, not proof the question is unanswerable |
| Answer generation | Question + context → structured answer | Answer, abstention, reason, cited chunk IDs/references | Returned in `RetrievalResult`; citations are constrained to returned chunk IDs |

The Hotpot adapter uses **fully isolated knowledge bases**: each `(context mode, question ID)` gets its own KB and vector directory. This fits this evaluator better than retrofitting scope through every existing entity/community model. It means the same context cannot be contaminated by another question even though the public query API still accepts optional scopes. Each indexing call is an explicit rebuild: JSONL is replaced and the KB's Chroma records are cleared first.

## Changes made

- `src/retrieval/query_planner.py`, `src/retrieval/community_retriever.py`: invalid relation hints no longer cause clarification; graph/vector/community errors are isolated, diagnostics are emitted, and community evidence honors chunk scope.
- `src/core/models.py`, `src/generation/answer_generator.py`, `src/engine.py`: structured answer and citation state is returned and synchronized; hybrid planning falls back safely; indexing returns graph/extraction statistics and stage timings and recovers from per-chunk extraction errors.
- `src/storage/vector_store.py`, `src/engine.py`: full reindex removes stale KB vector records before deterministic upserts.
- `configs/domains/academic.yaml`: adds defined types and directed relation semantics for film, literature, geographic, organization, and event relations, including `ACTED_IN`.
- `run_hotpot.py`: oracle and full-context selection are separate; each question/mode is isolated; all four retrieval routes run on the same indexed corpus; structured per-question failures, timings, route evidence, metrics, and route contributions are written to JSON.
- `src/extraction/schemas.py`: records prompt sizes/request timing and distinguishes timeout, connection, HTTP, malformed JSON, and non-transient errors; only transient or malformed responses retry, with bounded backoff.
- `tests/test_hotpot_repairs.py`: verifies oracle/full selection, invalid-plan fallback, route failure isolation, two-document graph traversal/provenance/context, and explicit vector rebuild cleanup.

## Performance and evaluation limits

Indexing reports loading, chunking, entity extraction, relationship extraction, entity resolution, embedding/indexing, graph construction/validation, community detection, summarization, total time, graph metrics, and LLM call counts. Query records report per-route total latency, LLM request counts, provenance checks, route candidates, and route errors; finer query retrieval substage latencies are not yet emitted. The evaluator computes macro supporting-fact recall, complete supporting-fact coverage, answer EM/F1, abstention, citation coverage, invalid-plan rate, provenance validity, failure count, mean/median latency, per-question route contributions, and Recall@5 grouped by benchmark question type. It does not claim any HotpotQA result until a live evaluation is run.

Extraction result caching and embedding reuse across separate engine processes are not implemented. Rebuilds are safe and idempotent but repeat inference and embedding. Ollama remains sequential, with bounded retry and per-chunk recovery; no concurrency is added. Graph path retrieval tests use controlled graph objects and synthetic facts, while actual Hotpot graph quality still depends on local extraction quality and needs the full-context evaluation output.

## Running diagnostics

```bash
# Offline tests (no Ollama required)
./.venv/bin/pytest -q

# One-question full-context four-route diagnostic
./.venv/bin/python run_hotpot.py --context-mode full --limit 1 --output experiments/hotpotqa/diagnostic_one.json

# Five-question smoke evaluation
./.venv/bin/python run_hotpot.py --context-mode full --limit 5

# Oracle-context evaluation, reported to a separate output file
./.venv/bin/python run_hotpot.py --context-mode oracle --limit 5

# Full-context evaluation
./.venv/bin/python run_hotpot.py --context-mode full --limit 20
```

The one-question runner reports `VECTOR_ONLY`, `GRAPH_ONLY`, `HYBRID`, and `COMMUNITY_AWARE` against the same question KB. This requires the configured local Ollama models; ordinary tests use no live service.
