# Hybrid GraphRAG Public Python API

This document provides a comprehensive guide to the public Python interface for the **Hybrid GraphRAG** engine.

The public interface is designed as an enterprise-ready, thin facade (`HybridGraphRAG`) exposing document ingestion, multi-tenancy knowledge-base isolation, multi-route retrieval, evidence-grounded answer generation, and verified citation validation.

---

## 1. Quick Start

### Installation & Import

Import the facade and response models directly from `hybrid_graphrag`:

```python
from hybrid_graphrag import (
    HybridGraphRAG,
    QueryResult,
    Citation,
    HybridGraphRAGError,
)
```

*(Note: Dual-path importing via `from src import HybridGraphRAG` is also fully supported for backwards compatibility).*

### Basic Usage

```python
from pathlib import Path
from hybrid_graphrag import HybridGraphRAG

# Initialize an isolated knowledge base
with HybridGraphRAG.from_config("configs/default.yaml", kb_id="medical_kb") as engine:
    # 1. Ingest files or directory
    engine.ingest("./data/medical_papers")

    # 2. Query the engine
    result = engine.query(
        question="What treatments are effective for severe asthma?",
        route="hybrid"
    )

    # 3. Access verified answer and citations
    print(result.answer)
    for c in result.citations:
        print(f"[{c.citation_id}] Document {c.doc_id}: {c.text}")
```

---

## 2. Factory Methods & Initialization

The `HybridGraphRAG` class provides two primary factory constructors to create or resume an engine instance.

### `HybridGraphRAG.from_config(...)`

Initializes a new knowledge base or attaches to an existing one using an explicit configuration dictionary or YAML file path.

```python
engine = HybridGraphRAG.from_config(
    config="configs/default.yaml",  # str, Path, or dict
    kb_id="research_kb",            # Knowledge base identifier
    domain_ontology_path=None       # Optional custom domain ontology YAML
)
```

- **`config`** (`str | Path | dict`): Path to a configuration YAML file or a dictionary conforming to `SystemConfig`.
- **`kb_id`** (`str`): Target knowledge base identifier. If omitted, defaults to `"default_kb"`.
- **`domain_ontology_path`** (`str | Path | None`): Optional path to a domain ontology schema for targeted entity and relation extraction.

### `HybridGraphRAG.from_existing(kb_dir)`

Reopens an existing, persisted knowledge base without re-ingesting documents, re-extracting entities, or re-indexing embeddings.

```python
engine = HybridGraphRAG.from_existing("knowledge_bases/research_kb")
```

- Automatically recovers `config_snapshot.yaml` saved during previous runs.
- Loads existing NetworkX graph, document chunks, and vector store collections immediately.
- Raises `KnowledgeBaseNotFoundError` if the directory does not exist or lacks snapshot configuration.

---

## 3. Document Ingestion

```python
stats = engine.ingest(
    source="path/to/docs",  # Single file or directory path
    document_ids=None       # Optional custom IDs mapping
)
```

- Accepts a file path (`.txt`, `.md`, `.json`, `.pdf`) or directory containing documents.
- Chunks text according to chunking configuration, runs entity & relation extraction, resolves coreferences and canonical entities, builds the knowledge graph, and indexes embeddings into ChromaDB.
- **Returns**: Dictionary with ingestion metrics (e.g. `{"document_count": 5, "chunk_count": 42}`).
- **Errors**: Raises `IngestionError` if the path does not exist or parsing fails.

---

## 4. Querying & Retrieval Routes

```python
result: QueryResult = engine.query(
    question="Were Scott Derrickson and Ed Wood of the same nationality?",
    route="hybrid",
    top_k=8
)
```

### Available Routes:
- `"hybrid"` *(Default)*: Reciprocal Rank Fusion (RRF) combining vector similarity and multi-hop graph traversal.
- `"vector"`: Dense semantic retrieval using chunk embeddings.
- `"graph"`: Multi-hop graph entity traversal over canonical entities and relations.
- `"community"`: Leiden community detection and graph-grounded community report retrieval.

---

## 5. Response Models

### `QueryResult`

The query method returns a structured `QueryResult` dataclass:

| Field | Type | Description |
|---|---|---|
| `question` | `str` | The input user query. |
| `answer` | `str` | Synthesized answer grounded strictly in verified evidence. |
| `route` | `str` | Retrieval route used (`"hybrid"`, `"vector"`, `"graph"`, `"community"`). |
| `citations` | `list[Citation]` | Verified supporting citations linked to source chunks. |
| `raw_context` | `str` | Full structured context provided to the generation LLM. |
| `abstained` | `bool` | `True` if the engine abstained due to insufficient evidence or conflict. |
| `abstention_reason` | `str | None` | Reason for abstention if `abstained=True`. |
| `latency_seconds` | `float` | Query latency in seconds. |
| `metadata` | `dict` | Diagnostic metrics (e.g. candidate entity counts, chunk IDs). |

### `Citation`

Each citation represents an evidence attribution verified against the underlying retrieved text chunks:

| Field | Type | Description |
|---|---|---|
| `citation_id` | `str` | Citation identifier in the answer (e.g., `"[1]"` or `"[CHK_001]"`). |
| `chunk_id` | `str` | Ground truth chunk ID from the knowledge base. |
| `doc_id` | `str` | Source document ID containing the chunk. |
| `text` | `str` | The exact supporting text passage. |
| `valid` | `bool` | Whether the citation was validated by the engine (`True`). |

---

## 6. Knowledge Base Isolation & Multi-Tenancy

Every knowledge base managed by `HybridGraphRAG` is strictly isolated within its own directory under `storage_root`:

```
knowledge_bases/
├── research_papers/
│   ├── config_snapshot.yaml
│   ├── graph.json
│   ├── chunks.json
│   └── chromadb/
└── medical_records/
    ├── config_snapshot.yaml
    ├── graph.json
    ├── chunks.json
    └── chromadb/
```

- **Zero Leakage**: Graph traversals, ChromaDB collections, and community reports are partitioned per `kb_id`.
- **Independent Lifecycles**: Knowledge bases can be ingested, queried, or deleted independently without affecting peers.

---

## 7. Lifecycle & Resource Management

Engines should be properly closed to release locks and underlying storage connections.

### Using Context Manager (Recommended):

```python
with HybridGraphRAG.from_config("configs/default.yaml", kb_id="my_kb") as engine:
    result = engine.query("What is the main finding?")
```

### Manual Lifecycle:

```python
engine = HybridGraphRAG.from_config("configs/default.yaml", kb_id="my_kb")
try:
    result = engine.query("What is the main finding?")
finally:
    engine.close()
```

---

## 8. Exception Hierarchy

All public exceptions inherit from `HybridGraphRAGError`:

```text
HybridGraphRAGError
├── ConfigurationError
├── KnowledgeBaseNotFoundError
├── IngestionError
├── QueryError
└── ModelError
```

Catch them cleanly in client code:

```python
from hybrid_graphrag import HybridGraphRAG, ConfigurationError, KnowledgeBaseNotFoundError

try:
    engine = HybridGraphRAG.from_existing("non_existent_kb")
except KnowledgeBaseNotFoundError as e:
    print(f"KB directory not found: {e}")
except ConfigurationError as e:
    print(f"Invalid configuration: {e}")
```
