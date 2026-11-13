# Implementation Log (`docs/implementation_log.md`)

## Phase 0 — Environment, Model & Architecture Audit (Completed 2026-11-13)
- **Status**: Completed and verified.
- **Verified Environment**: Python 3.13.7, Git 2.54.0, GitHub CLI 2.103.3, Ollama 0.40.16.
- **Verified Models**: `gemma3:4b` (3.4 GB) and `embeddinggemma:latest` (621 MB, 768-d output verified).
- **Repository**: Initialized fresh Git repo on branch `hybrid-graphrag-build`.
- **Next Step**: Phase 1, Step 1 — Configuration loading/validation, Ontology enforcement, and Core Pydantic models/interfaces.

## Phase 1, Step 1 — Configuration, Core Models & Ontology Enforcement (Completed 2026-11-13)
- **Status**: Completed and verified.
- **Implemented Modules**:
  - `src/core/models.py`: 19-stage pipeline provenance models (`Document`, `Page`, `Chunk`, `EntityMention`, `CanonicalEntity`, `Relationship`, `CommunityReport`, `QueryPlan`, `Evidence`, `RetrievalResult`).
  - `src/core/ontology.py`: Dynamic ontology models (`EntityTypeDefinition`, `RelationshipTypeDefinition`, `DomainOntology`) supporting YAML-driven validation.
  - `src/core/interfaces.py`: Clean typing protocol interfaces separating core from concrete implementations.
  - `src/core/configuration.py`: YAML loading, deep dictionary merging, Pydantic validation, and deterministic SHA-256 configuration hashing.
  - `configs/`: Domain YAML configs for Academic, Legal, Biomedical, and Financial domains plus `default.yaml`.
- **Test Verification**: 4 tests passing in `tests/test_core_config_and_models.py`.

## Phase 1, Step 2 & Step 3 — Ingestion, Chunking & JSONL DocumentStore (Completed 2026-11-13)
- **Status**: Completed and verified.
- **Implemented Modules**:
  - `src/ingestion/pdf_loader.py` (`PDFLoader`):
    - Satisfies `DocumentLoader` protocol.
    - Deterministically sorts files by filename and assigns sequential IDs (`D001`, `D002`, ...) scoped by `kb_id`.
    - Supports `.pdf` (using `pypdf.PdfReader`), `.txt`, and `.md` (supporting single-page and `\f` form-feed splitting).
    - Per-page text extraction, `\r\n` to `\n` normalization, line-by-line and trailing whitespace stripping, SHA-256 `content_hash`.
    - Automatic detection of empty pages (`Page.is_empty = True`, appended to `Document.empty_pages`).
    - Identification of scanned/empty documents (`Document.is_scanned_or_empty = True`) with warning logging.
  - `src/ingestion/chunker.py` (`DeterministicChunker`):
    - Satisfies `TextChunker` protocol and accepts `ChunkingConfig` (`chunk_size=1100`, `chunk_overlap=180`, `respect_sentence_boundaries=True`).
    - Groups non-empty pages by `doc_id` in ascending `page_number` order, tracking exact `(start_char, end_char, page_number)` spans.
    - Boundary preference hierarchy: paragraph (`\n\n`) -> newline (`\n`) -> sentence punctuation (`. `, `? `, `! `) within the last 25% of the chunk window before falling back to hard character cuts.
    - Sliding window advancement using `chunk_overlap` (180 chars) with strict forward movement guarantees preventing infinite loops.
    - Deterministic chunk IDs (`f"{doc_id}_C{chunk_index:04d}"`), SHA-256 `content_hash`, and cross-page provenance tracking (`page_numbers=[1, 2]`).
  - `src/storage/document_store.py` (`DocumentStore`):
    - Atomic JSONL persistence and loading for `Document` (`documents.jsonl`), `Page` (`pages.jsonl`), and `Chunk` (`chunks.jsonl`) under `knowledge_bases/<kb_id>/data/processed/`.
    - Strict `kb_id` isolation enforcement on save, load, and query operations.
    - O(1) in-memory lookups: `get_chunk(chunk_id)`, `get_chunks_by_ids(chunk_ids)`, and `list_all_chunks()`.
    - Document and page queries: `get_document(doc_id)`, `list_all_documents()`, `get_pages_for_document(doc_id)`.
  - `tests/test_ingestion_and_docstore.py`:
    - 11 comprehensive unit tests covering reportlab-free programmatic multi-page PDF generation via `pypdf`, blank/scanned page detection, deterministic sorting, synthetic form-feed text, sentence boundary hierarchy, cross-page provenance, JSONL round-trip, and KB isolation.
- **Verification Results**:
  - Full test suite: 15 passing tests (`./.venv/bin/pytest -v` executed in 0.20s).
- **Next Step**: Phase 1, Step 4 (Entity Extraction) & Step 5 (Relationship Extraction).

## Phase 1, Step 4 & Step 5 — Entity & Relationship Extraction with Strict Validation (Completed 2026-11-13)
- **Status**: Completed and verified.
- **Implemented Modules**:
  - `prompts/entity_extraction.txt` & `prompts/relationship_extraction.txt`:
    - Clean domain-agnostic prompt templates using `{entity_types}`, `{relationship_types}`, `{chunk_text}`, and `{mentions_json}` placeholders.
    - Strict instruction for exact verbatim `evidence_quote` extraction from chunk text and explicit polarity specification (`"POSITIVE"`, `"NEGATIVE"`, `"UNCERTAIN"`).
  - `src/extraction/schemas.py`:
    - Raw LLM Pydantic schemas: `RawEntityMentionItem`, `RawEntityExtractionOutput`, `RawRelationshipItem`, `RawRelationshipExtractionOutput`.
    - `OllamaLLMClient`: REST client for Ollama `/api/generate` with JSON formatting, bounded retries on malformed JSON or validation errors, and mock support via `custom_generator`.
  - `src/extraction/entity_extractor.py` (`OllamaEntityExtractor`):
    - Satisfies `EntityExtractor` protocol.
    - Surface name normalization to `normalized_name` (lowercase, whitespace collapse, punctuation stripping).
    - Character offset detection (`char_start`, `char_end`) via exact and case-insensitive substring matching in `chunk.text`.
    - Strict ontology validation using `DomainOntology.is_valid_entity_type()` with warning logs on unrecognized types.
    - Deterministic mention IDs (`f"{chunk.chunk_id}_M{idx:03d}"`) preserving chunk, document, and page provenance.
  - `src/extraction/relation_extractor.py` (`OllamaRelationshipExtractor`):
    - Satisfies `RelationshipExtractor` protocol.
    - Flexible entity mention resolution by mention ID or surface name, with self-loop elimination.
    - Three-stage validation pipeline setting `Relationship.validation_status`:
      1. Quote Verification: Verbatim case-insensitive substring check (`REJECTED_QUOTE_MISMATCH` if missing or fabricated).
      2. Negation & Uncertainty Filtering: Detects negative/uncertain polarity or explicit negation keywords (`REJECTED_NEGATION`).
      3. Ontology Validation: Enforces source/target endpoint type constraints via `ontology.validate_triple_types()` (`REJECTED_ONTOLOGY`).
      4. Fully validated relationships assigned `ValidationStatus.VALIDATED`.
  - `src/storage/document_store.py`:
    - Extended with `save_entity_mentions`, `load_entity_mentions`, `save_triples`, and `load_triples` persisting to `entity_mentions.jsonl` and `triples.jsonl` under `knowledge_bases/<kb_id>/data/processed/`.
    - Strict `kb_id` isolation enforcement across all operations.
  - `tests/test_extraction.py`:
    - 5 comprehensive unit tests covering surface name normalization, teaching example entity extraction (`Rahul`, `IIIT Naya Raipur`, `Professor Sharma`), bounded LLM client retries, all three relationship validation rejection statuses, and JSONL persistence round-trips.
- **Verification Results**:
  - Full test suite: 20 passing tests (`./.venv/bin/pytest -v` executed in 0.22s).
- **Next Step**: Phase 1, Step 6 (Entity Resolution) & Step 7 (Embedding Service).

## Phase 1, Step 6 & Step 7 — Embedding Service & Conservative Entity Resolution (Completed 2026-11-13)
- **Status**: Completed and verified.
- **Implemented Modules**:
  - `src/embeddings/embedding_service.py` (`OllamaEmbeddingService`):
    - Satisfies `EmbeddingProvider` protocol (`embed_text` and `embed_batch`).
    - Interfaces with Ollama's `/api/embed` endpoint with `embeddinggemma:latest` (768-d).
    - Supports `custom_embedder` callable injection for fast, offline, deterministic unit testing.
    - Strict vector validation: enforces `expected_embedding_dim` (768-d), verifies all floats are finite (no NaN or Inf), and verifies non-zero L2 norm (> 1e-12).
    - `cosine_similarity_matrix(a, b)`: NumPy vectorized L2-normalized cosine similarity calculation.
  - `prompts/entity_resolution.txt`:
    - Strict, conservative resolution prompt template accepting `{entity_a_json}` and `{entity_b_json}`.
    - Instructs the model to output `{"is_same_entity": bool, "reason": str}` only when unambiguous evidence demonstrates identity.
  - `src/resolution/entity_resolver.py` (`ConservativeEntityResolver`):
    - Implements the complete 5-step resolution pipeline:
      - **17A (Normalization & Honorifics)**: Surface name normalization and title stripping (`Prof.`, `Professor`, `Dr.`, `Doctor`, `Mr.`, `Ms.`) for `PERSON` entities alongside `DomainOntology.static_aliases`.
      - **17B (Exact & Alias Matching)**: Checks exact normalized names, existing aliases, static aliases, and honorific core token matches with `require_type_match`.
      - **17C (Embedding Discovery)**: Discovers candidate canonical entities of matching type via cosine similarity on descriptive representation text.
      - **17D (Verification & Ambiguity Band)**: Auto-merges when similarity $\ge 0.95$; invokes LLM verification when $0.82 \le \text{sim} < 0.95$ with `EntityResolutionVerdict` validation, defaulting to conservative separation.
      - **17E (Canonical Creation & Provenance Merging)**: Assigns sequential IDs (`E001`, `E002`, ...), preserves aliases, source mention IDs, chunk IDs, doc IDs, and concatenates unique descriptions.
    - `resolve_and_remap`: Remaps validated relationships from mention IDs to canonical entity IDs, eliminates self-loops, and aggregates duplicate edges sharing `(source, relation_type, target, scope_id)` with combined provenance and `support_count = len(source_chunk_ids)`. Keeps distinct relation types separate.
  - `src/storage/document_store.py`:
    - Extended with `save_mention_to_entity` & `load_mention_to_entity` (`knowledge_bases/<kb_id>/data/processed/mention_to_entity.jsonl`).
    - Extended with `save_canonical_entities` & `load_canonical_entities` (`knowledge_bases/<kb_id>/graph/entities.jsonl`).
    - Extended with `save_relationships` & `load_relationships` (`knowledge_bases/<kb_id>/graph/relationships.jsonl`).
    - Extended `save_all` to support atomic persistence across all processed and graph layers.
  - `tests/test_embeddings_and_resolution.py`:
    - 8 comprehensive tests covering vector dimension/finiteness/norm validation, cosine similarity matrix computation, exact & honorific alias resolution (`Professor Sharma`, `Prof. Sharma`, `Dr. Sharma` $\to$ single `E002`), ambiguous similarity band with LLM merge and rejection, relationship endpoint remapping/aggregation, self-loop dropping, and DocumentStore graph persistence.
- **Verification Results**:
  - Full test suite: 27 passed, 1 skipped (live Ollama test gracefully skipped when offline) in 0.40s.
- **Next Step**: Phase 1, Steps 8, 10, 11, 12, and 13 (VectorStore, igraph Builder, Graph Validator, Leiden Clustering, Community Summarizer).

## Phase 1, Steps 8, 10, 11, 12 & 13 — Offline Indexing Pipeline Completion (Completed 2026-11-13)
- **Status**: Completed and verified.
- **Implemented Modules**:
  - `src/storage/vector_store.py` (`ChromaVectorStore`):
    - Scoped `chromadb.PersistentClient` per `kb_id` under `knowledge_bases/<kb_id>/vector_store/`.
    - Manages 3 mandatory collections with cosine metric (`metadata={"hnsw:space": "cosine"}`):
      1. `entity_descriptions`: Upserts canonical entity representations and queries with optional `entity_type` filtering.
      2. `text_chunks`: Upserts text chunks and queries with optional `scope_id` filtering.
      3. `community_reports`: Upserts synthesized community reports and queries by topic embedding.
    - Always stores and enforces `kb_id` isolation across all upsert and query methods.
  - `src/graph/graph_builder.py` (`IGraphBuilder`):
    - Builds a directed `igraph.Graph(directed=True)` from canonical entities and validated relationships.
    - Vertices sorted by `entity_id` with attributes: `name`, `entity_id`, `label`, `entity_type`, `description`, `aliases`, `source_chunk_ids`, `kb_id`.
    - Directed edges preserve multigraph semantics: distinct `relation_type` edges between the same node pair remain separate, while duplicate identical edges are aggregated with merged supporting chunk IDs and quotes.
    - `reconstruct_from_store(doc_store)`: Directly reconstructs the graph from `entities.jsonl` and `relationships.jsonl`.
  - `src/graph/graph_validator.py` (`GraphValidator`):
    - Validates graph structural integrity: confirms all edge endpoints exist in vertices/entities, relation types comply with `DomainOntology`, and chunk IDs exist in `valid_chunk_ids`.
    - Computes structural metrics: `node_count`, `edge_count`, `connected_components_count`, `isolated_nodes`, `relation_type_distribution`.
    - Generates deterministic SHA-256 `graph_fingerprint` over canonical sorted vertex and edge tuples.
    - Emits and saves `manifest.json` under `knowledge_bases/<kb_id>/manifest.json`.
  - `src/graph/clustering.py` (`LeidenCommunityDetector`):
    - Keeps the original directed graph strictly unchanged.
    - Constructs an undirected projection `G_undirected = igraph.Graph(directed=False)` with identical vertices.
    - Sets undirected edge weights to `len(distinct supporting source_chunk_ids)` across all directed edges between each unordered pair $\{u, v\}$.
    - Partitions via `leidenalg.find_partition(G_undirected, leidenalg.RBConfigurationVertexPartition, weights="weight", resolution_parameter=res, seed=seed)`.
    - Saves communities with metadata (`community_id`, `member_entity_ids`, `size`, `partition_id`, `resolution`, `seed`, `objective_value`, `graph_fingerprint`) to `knowledge_bases/<kb_id>/graph/communities.jsonl`.
  - `prompts/community_summary.txt`:
    - Clean prompt template accepting `{community_id}`, `{entities_json}`, `{relationships_json}`, and `{supporting_chunks_text}` returning strict JSON `{"summary_text": "..."}`.
  - `src/communities/community_summarizer.py` (`CommunitySummarizer`):
    - Generates grounded `CommunityReport` records using member vertices, internal edges, and chunk evidence.
    - Persists reports to `knowledge_bases/<kb_id>/graph/community_reports.jsonl`.
    - Embeds report summaries and indexes them into `ChromaVectorStore`'s `community_reports` collection.
  - `src/storage/document_store.py`:
    - Extended with `save_communities`, `load_communities`, `save_community_reports`, and `load_community_reports`.
    - `save_all` updated to support all processed, graph, community, and report records.
  - `tests/test_vector_graph_and_communities.py`:
    - 4 comprehensive unit tests verifying ChromaDB collections/queries (including `entity_type` and `scope_id` filtering), teaching graph construction/multigraph preservation/manifest generation/store reconstruction, Leiden clustering weight calculation/directed graph immutability, and community summarizer generation/vector indexing.
- **Verification Results**:
  - Full test suite: 31 passed, 1 skipped in 0.99s (`./.venv/bin/pytest -v`).
- **Next Step**: Phase 2 — Online Query Engine (Query Planner, Local Search, Global Search, Mixed Search, Hybrid RRF, Answer Generator).


## Phase 1, Steps 14 through 23 — Online Query Engine, Hybrid Retrieval & Answer Generation (Completed 2026-11-13)
- **Status**: Completed and verified.
- **Implemented Modules**:
  - `prompts/query_planning.txt`:
    - Structured prompt accepting `{question}`, `{entity_types}`, and `{relationship_types}` placeholders.
    - Guides the LLM to output strict JSON conforming to `QueryPlan` (`intent`, `seed_entities`, `target_type`, `relationship_path`, `requires_clarification`, `clarification_reason`).
  - `prompts/answer_generation.txt`:
    - Strict answer generation prompt accepting `{question}` and `{formatted_context}`.
    - Enforces factual grounding, mandatory inline citations `[chunk_id, doc_id, p.X]`, conflicting evidence reporting, and clean abstention handling when context is insufficient.
  - `src/retrieval/query_planner.py` (`OllamaQueryPlanner`):
    - Converts natural language questions into validated `QueryPlan` instances via `OllamaLLMClient`.
    - Validates relationship types against `DomainOntology.is_valid_relation_type()`, stripping non-permitted types and setting `requires_clarification=True` if necessary.
  - `src/retrieval/seed_retriever.py` (`SeedEntityRetriever`):
    - Resolves natural language `SeedEntitySpec` items to canonical `entity_id`s using a conservative 4-step hierarchy:
      1. Exact normalized canonical name match (with optional entity type constraint).
      2. Known alias and honorific-stripped prefix matching (`Dr.`, `Prof.`, etc.).
      3. Embedding-based candidate discovery in `ChromaVectorStore` (`entity_descriptions` collection) filtered by similarity threshold ($\ge 0.82$).
      4. Ambiguity detection: flags ties within a 0.02 similarity margin.
  - `src/graph/traversal.py` (`GraphTraverser`) & `src/retrieval/graph_retriever.py` (`GraphEvidenceRetriever`):
    - `GraphTraverser`: Executes directed BFS traversal up to `max_hops` (3), prevents vertex cycles per path, enforces `scope_id` edge isolation, and prioritizes paths aligned with `QueryPlan.relationship_path`.
    - `GraphEvidenceRetriever`: Aggregates traversed paths into supporting chunks, sorts deterministically by `(best_alignment DESC, min_hop ASC, support_count DESC, chunk_id ASC)`, assigns `graph_rank = 1, 2, ...`, and emits `Evidence` objects tagged with `ProvenanceRoute.GRAPH`.
  - `src/retrieval/vector_retriever.py` (`VectorEvidenceRetriever`):
    - Embeds questions using `EmbeddingProvider` and retrieves top-$k$ chunks from `ChromaVectorStore` (`text_chunks` collection) with optional `scope_id` filtering.
    - Assigns `vector_rank = 1, 2, ...` and emits `Evidence` objects tagged with `ProvenanceRoute.VECTOR`.
  - `src/retrieval/evidence_merger.py` (`EvidenceMerger`):
    - Merges multiple evidence streams (`GRAPH`, `VECTOR`, `COMMUNITY`) by `chunk_id`.
    - Computes union of provenance routes, preserves best (minimum) ranks, and merges supporting graph paths and edge IDs without duplication.
  - `src/retrieval/reranker.py` (`RRFReranker`):
    - Computes Reciprocal Rank Fusion: $\text{score} = \frac{1}{k + \text{graph\_rank}} + \frac{1}{k + \text{vector\_rank}}$ with $k = 60$.
    - Implements deterministic multi-key tie-breaking: `(-rrf_score, -len(supporting_edge_ids), has_graph_rank, chunk_id)`.
  - `src/retrieval/community_retriever.py` (`CommunityRetriever` & `HybridRetrieverOrchestrator`):
    - `CommunityRetriever`: Embeds questions and retrieves matching `CommunityReport` records from ChromaDB, collecting their supporting chunks as `Evidence` with `ProvenanceRoute.COMMUNITY`.
    - `HybridRetrieverOrchestrator`: Dispatches queries by `QueryIntent`:
      - `LOCAL`: Resolves seeds $\to$ retrieves graph + vector evidence $\to$ merges $\to$ RRF reranks.
      - `GLOBAL`: Retrieves community reports + vector evidence $\to$ merges $\to$ RRF reranks.
      - `MIXED`: Resolves seeds $\to$ retrieves graph + community + vector evidence $\to$ merges $\to$ RRF reranks.
      - Emits validated `RetrievalResult` including abstention status and reasons.
  - `src/retrieval/context_builder.py` (`ContextBuilder`):
    - Assembles Community Summaries, Graph Traversal Paths, and Primary Evidence Chunks into structured Markdown under a hard character budget (`max_context_chars=12000`).
  - `src/generation/answer_generator.py` (`OllamaAnswerGenerator`):
    - Generates strictly grounded answers with citations `[chunk_id, doc_id, p.X]`.
    - Handles abstentions cleanly when `retrieval_result.abstained` is `True` or when no relevant evidence exists.
  - `tests/test_online_retrieval_and_generation.py`:
    - 7 comprehensive unit tests verifying query planning with ontology validation, seed resolution with honorifics/aliases/ties, directed cycle-free graph traversal, evidence merging, RRF mathematical accuracy, scoped vector retrieval, and end-to-end local/global/mixed orchestration and answer generation.
- **Verification Results**:
  - Full test suite: 38 passed, 1 skipped in 0.98s (`./.venv/bin/pytest -v`).
- **Next Step**: Phase 3 — End-to-End CLI & Evaluation Harness.


## Phase 1, Steps 24, 25 & 26 — End-to-End Orchestrator, CLI Entrypoint, Multi-Domain Isolation & Live Ollama Smoke Run (Completed 2026-11-13)
- **Status**: Completed and verified.
- **Implemented Modules**:
  - `src/engine.py` (`HybridGraphRAGEngine`):
    - Unified 19-stage engine initializing isolated KB directory trees, document stores, ChromaDB collections, and graph caches.
    - `index_directory(input_dir)`: Fully automates Stages 1–10: raw loading $\to$ chunking $\to$ ontology-bound entity/relation extraction $\to$ conservative entity resolution & remapping $\to$ ChromaDB embeddings $\to$ igraph multigraph construction & validation (`manifest.json`) $\to$ Leiden community detection $\to$ community summarization & indexing $\to$ JSONL persistence.
    - `query(question, mode, scope_id)`: Implements Stages 11–19 supporting `hybrid` (full pipeline), `vector_only`, `graph_only`, and `community_aware` modes.
  - `main.py` & `src/main.py`:
    - CLI entrypoints supporting `--config`, `--index-dir`, `--query`, `--mode`, and `--scope-id`.
  - `tests/test_end_to_end_and_isolation.py`:
    - Tests complete 19-stage pipeline on `academic_kb` and `company_kb`.
    - Validates all 5 required question types (single-hop, 3-hop traversal, broad global theme, alias query with honorifics, unanswerable query with clean abstention).
    - Validates strict zero cross-KB leakage between `academic_kb` and `company_kb`.
  - `experiments/run_phase1_smoke.py`:
    - Live local Ollama execution script indexing real academic and company corpora with `gemma3:4b` and `embeddinggemma:latest`, saving full execution telemetry to `experiments/self_generated/phase1_smoke_results.json`.
  - `docs/PHASE1_HYBRID_GRAPHRAG.md`, `docs/architecture.md`, `README.md`:
    - Comprehensive system architecture specifications, storage layout diagrams, CLI usage instructions, and Phase 1 verification summary.
- **Verification Results**:
  - Full test suite: 40 tests, 100% pass rate in < 1.0s (`./.venv/bin/pytest -v`).
