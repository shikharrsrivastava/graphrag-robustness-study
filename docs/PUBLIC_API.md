# Hybrid GraphRAG Public Python API Reference

This document provides a complete reference for the public Python interface of the **Hybrid GraphRAG** engine.

The public interface is designed as an enterprise-grade facade (`HybridGraphRAG`) providing document ingestion, multi-tenant knowledge base isolation, multi-route retrieval, evidence-grounded answer generation, and verifiable citation validation.

---

## 1. Quick Start

### Installation & Import

Import the primary facade, response models, and exception classes from `hybrid_graphrag`:

```python
from hybrid_graphrag import (
    HybridGraphRAG,
    QueryResult,
    Citation,
    HybridGraphRAGError,
    ConfigurationError,
    IngestionError,
    KnowledgeBaseNotFoundError,
    QueryError,
)
```

### Complete End-to-End Example

```python
from pathlib import Path
from hybrid_graphrag import HybridGraphRAG

# 1. Initialize a new knowledge base from configuration
with HybridGraphRAG.from_config("configs/default.yaml", kb_id="medical_research") as engine:
    # 2. Ingest document directory (PDFs, Markdown, text)
    stats = engine.ingest("./data/medical_papers", rebuild=True)
    print(f"Ingested {stats.get('documents_processed')} documents into {engine.kb_id}")

    # 3. Query the knowledge base
    result = engine.query(
        question="What treatments are effective for severe asthma?",
        mode="hybrid"  # Options: "hybrid", "vector_only", "graph_only", "community_aware"
    )

    # 4. Access structured answer, canonical answer, and citations
    print(f"Canonical Answer: {result.canonical_answer}")
    print(f"Answer: {result.answer}\n")
    print("Verified Citations:")
    for citation in result.citations:
        print(f"- [{citation.chunk_id}] {citation.filename} (p.{citation.page_numbers}): {citation.text_snippet}")
```

---

## 2. Factory Methods & Initialization

The `HybridGraphRAG` class provides two primary constructors to create or resume an engine instance.

### `HybridGraphRAG.from_config(...)`

Initializes a new knowledge base or re-attaches to an existing one using an explicit configuration dictionary, domain YAML file, or `EngineConfig` object.

```python
engine = HybridGraphRAG.from_config(
    config="configs/default.yaml",  # Union[str, Path, Dict[str, Any], EngineConfig]
    kb_id="research_kb",            # Optional[str] override for knowledge base ID
    base_dir=".",                   # Union[str, Path] base directory for storage
    llm_client=None,                # Optional custom OllamaLLMClient
    embedding_service=None,         # Optional custom EmbeddingProvider
)
```

- **`config`** (`str | Path | dict | EngineConfig`): Path to a configuration YAML (or domain configuration file), dictionary mapping, or validated `EngineConfig`.
- **`kb_id`** (`str | None`): Optional knowledge base identifier override. Must not contain path separators.
- **`base_dir`** (`str | Path`): Base directory for resolving relative storage paths (default: `"."`).
- **`llm_client`** (`OllamaLLMClient | None`): Optional custom LLM client for non-Ollama inference or testing mocks.
- **`embedding_service`** (`EmbeddingProvider | None`): Optional custom dense embedding provider.

### `HybridGraphRAG.from_existing(...)`

Reopens an existing, already-indexed knowledge base without re-ingesting documents, re-extracting entities, or re-computing embeddings.

```python
engine = HybridGraphRAG.from_existing(
    kb_dir="knowledge_bases/research_kb",  # Union[str, Path]
    base_dir=".",                          # Union[str, Path]
    llm_client=None,                       # Optional custom OllamaLLMClient
    embedding_service=None,                # Optional custom EmbeddingProvider
)
```

- Automatically loads `config_snapshot.yaml` saved during original indexing.
- Re-hydrates in-memory `igraph.Graph` from persisted JSONL records and attaches to existing ChromaDB vector collections.
- Raises `KnowledgeBaseNotFoundError` if the directory does not exist or lacks `config_snapshot.yaml`.

---

## 3. Core Engine Methods

### `engine.ingest(...)`

Ingests, chunks, extracts entities/relations, builds knowledge graphs, runs Leiden clustering, and indexes documents.

```python
stats = engine.ingest(
    input_path="./inputs/my_docs",  # Union[str, Path] directory or PDF file
    rebuild=True                    # bool, whether to clear existing records
)
```

- **`input_path`**: Path to a directory containing `.pdf`, `.txt`, or `.md` files, or a single PDF file.
- **`rebuild`**: If `True` (default), clears any previous index for this `kb_id` before indexing.
- **Returns**: Dictionary of indexing metrics (e.g. `documents`, `chunks`, `entities`, `relationships`, `communities`).
- **Raises**: `IngestionError` if the input path does not exist or parsing fails.

### `engine.query(...)`

Executes multi-route retrieval, reciprocal rank fusion, and grounded answer generation with verifiable inline citations.

```python
result: QueryResult = engine.query(
    question="Which writer was from England, Henry Roth or Robert Erskine Childers?",
    mode="hybrid",        # "hybrid" | "vector_only" | "graph_only" | "community_aware"
    route=None,           # Backward-compatible alias for 'mode'
    scope_id=None,        # Optional tenant or document scope filter
)
```

- **`question`** (`str`): The natural language question to answer. Must not be empty.
- **`mode`** (`str`): Retrieval routing strategy.
  - `"hybrid"` *(Default)*: Reciprocal Rank Fusion (RRF $k=60$) combining multi-hop `igraph` BFS traversal and ChromaDB dense vector search.
  - `"vector_only"`: Dense cosine similarity retrieval over chunk embeddings.
  - `"graph_only"`: Seed-directed multigraph traversal over canonical entities and relations.
  - `"community_aware"`: Leiden community detection reports combined with local chunk evidence.
- **`route`** (`str | None`): Alias for `mode` for backward compatibility.
- **`scope_id`** (`str | None`): Optional tenant or document metadata filter.
- **Returns**: `QueryResult` object.
- **Raises**: `QueryError` if retrieval or generation fails; `ValueError` for invalid questions or modes.

### `engine.get_kb_stats()`

Returns runtime summary statistics of the active knowledge base:

```python
stats = engine.get_kb_stats()
```

Returns:
```python
{
    "kb_id": "medical_research",
    "documents": 12,
    "chunks": 148,
    "entities": 320,
    "relationships": 580,
    "communities": 18,
    "community_reports": 18,
    "vector_store_counts": {
        "chunks": 148,
        "entities": 320,
        "community_reports": 18
    },
    "graph": {
        "vertices": 320,
        "edges": 580
    }
}
```

### `engine.close()`

Releases cached graph memory structures, document index references, and underlying storage handles.

---

## 4. Response Data Models

### `QueryResult`

The structured result returned by `engine.query()`:

| Field | Type | Description |
| :--- | :--- | :--- |
| `question` | `str` | The original user question. |
| `answer` | `str` | Full synthesized answer with inline chunk citations (e.g. `[D007_C0000]`). |
| `canonical_answer` | `str \| None` | Concise, normalized direct answer (e.g. `"yes"`, `"no"`, entity name, or phrase). |
| `abstained` | `bool` | `True` if the engine abstained due to missing or conflicting evidence. |
| `abstention_reason` | `str \| None` | Explicit diagnostic explanation if `abstained=True`. |
| `citations` | `List[Citation]` | List of verified citations supporting the answer. |
| `cited_chunk_ids` | `List[str]` | Unique chunk identifiers cited in the answer. |
| `mode` | `str` | Active retrieval mode (e.g. `"hybrid"`). |
| `route` | `str` | Property alias for `mode`. |
| `latency_seconds` | `float` | Total query execution time in seconds. |
| `timing` | `Dict[str, float]` | Detailed latency breakdown (e.g. `query_planning_seconds`, `answer_generation_seconds`). |
| `raw_result` | `RetrievalResult \| None` | Internal retrieval result containing full evidence objects. |

### `Citation`

Represents an individual evidence citation validated against source document chunks:

| Field | Type | Description |
| :--- | :--- | :--- |
| `chunk_id` | `str` | Unique chunk identifier (e.g. `"D007_C0000"`). |
| `citation_id` | `str` | Property alias for `chunk_id` for backward compatibility. |
| `doc_id` | `str` | Source document identifier (e.g. `"D007"`). |
| `filename` | `str` | Source file name (e.g. `"ctx_006_Robert_Erskine_Childers.txt"`). |
| `page_numbers` | `List[int]` | Source page numbers where the passage appears. |
| `text_snippet` | `str \| None` | Excerpt of the cited chunk text. |
| `text` | `str \| None` | Property alias for `text_snippet`. |
| `valid` | `bool` | Whether the citation was verified against retrieved chunks (`True`). |

---

## 5. Lifecycle Management

Instances implement the Python context manager protocol (`__enter__` / `__exit__`):

```python
# Context manager automatically calls engine.close() upon exit
with HybridGraphRAG.from_existing("knowledge_bases/my_kb") as engine:
    result = engine.query("What is the primary conclusion?")
    print(result.answer)
```

Manual lifecycle pattern:
```python
engine = HybridGraphRAG.from_existing("knowledge_bases/my_kb")
try:
    result = engine.query("What is the primary conclusion?")
finally:
    engine.close()
```

---

## 6. Exception Hierarchy

All exceptions raised by the public API derive from `HybridGraphRAGError`:

```text
HybridGraphRAGError
├── ConfigurationError          # Invalid configuration format or missing files
├── KnowledgeBaseNotFoundError  # Target KB directory or snapshot does not exist
├── IngestionError              # Document loading, parsing, or indexing failures
└── QueryError                  # Retrieval, planning, or generation execution errors
```

Example exception handling:
```python
from hybrid_graphrag import HybridGraphRAG, ConfigurationError, KnowledgeBaseNotFoundError

try:
    engine = HybridGraphRAG.from_existing("knowledge_bases/nonexistent_kb")
except KnowledgeBaseNotFoundError as exc:
    print(f"KB not found: {exc}")
except ConfigurationError as exc:
    print(f"Configuration error: {exc}")
```

---

## 7. Retrieval Configuration & Provenance Models

### `RetrievalConfig` Configuration Parameters

Retrieval options configured in `configs/default.yaml` or programmatically via `EngineConfig.retrieval`:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_traversal_hops` | `int` | `3` | Maximum BFS hops on knowledge graph edges ($1..6$). |
| `max_graph_candidates` | `int` | `20` | Maximum candidate evidence chunks retrieved from the graph route. |
| `vector_top_k` | `int` | `10` | Top-$K$ candidate chunks retrieved from ChromaDB dense vector search. |
| `community_top_k` | `int` | `3` | Number of hierarchical community summaries retrieved for global queries. |
| `final_context_chunks` | `int` | `8` | Final chunk count budgeted for answer generation after RRF fusion. |
| `rrf_k` | `int` | `60` | Reciprocal Rank Fusion constant denominator factor. |
| `max_context_chars` | `int` | `12000` | Strict character window limit for LLM prompt context construction. |
| `provenance_expansion_enabled` | `bool` | `False` | Enable 1-hop Entity–Chunk–Entity bipartite provenance expansion. |
| `max_chunks_per_cooccurring_entity` | `int` | `2` | Hub suppression threshold: max chunks an entity can appear in before exclusion as a generic hub. |
| `max_cooccurring_entities_per_chunk`| `int` | `5` | Maximum co-occurring entities expanded per direct seed chunk. |
| `max_expanded_candidates_per_query` | `int` | `10` | Maximum total expanded chunk candidates per query. |
| `direct_seed_outranks_expanded` | `bool` | `True` | Direct seed chunks (`alignment=1.0`) strictly outrank expanded chunks (`alignment=0.85`). |

### Provenance Attribution Models

Retrieval evidence items explicitly distinguish between direct seed matches, traversed multigraph edges, and documentary co-occurrence:

- **`GraphProvenanceType`** (`Enum`):
  - `GRAPH_SEED_PROVENANCE`: Chunk directly contains a resolved query seed entity.
  - `GRAPH_EDGE_PROVENANCE`: Chunk supports an edge along a validated graph traversal path.
  - `ENTITY_CHUNK_COOCCURRENCE`: Chunk recovered through 1-hop documentary co-occurrence (`Seed -> Chunk A -> Cooccur Entity -> Chunk B`).

- **`CooccurrenceProvenance`** (`BaseModel`):
  - `seed_entity_id`: Resolved query seed entity ID.
  - `direct_chunk_id`: Direct chunk containing the seed entity.
  - `cooccurring_entity_id`: Entity co-occurring within the direct chunk.
  - `expanded_chunk_id`: Target chunk where the co-occurring entity also appears.
  - `expansion_depth`: Strictly fixed at `1` (one-hop constraint).
  - `skipped_hub_entities`: List of generic hub entity IDs pruned during expansion.
  - `fan_out_pruned`: Whether expansion was bounded by query fan-out limits.
