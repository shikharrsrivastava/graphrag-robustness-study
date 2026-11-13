# Hybrid GraphRAG System

A production-grade, offline-first **Hybrid GraphRAG** (Graph Retrieval-Augmented Generation) system built entirely on local models via **Ollama**.

The system merges the precision of directed knowledge graph traversal with the semantic breadth of dense vector search and macro-thematic Leiden community detection, fused through Reciprocal Rank Fusion (RRF) and validated against configurable domain ontologies.

---

## Key Capabilities

- **19-Stage End-to-End Pipeline**: Complete pipeline from raw multi-page PDF ingestion and boundary-aware deterministic chunking to ontology-constrained extraction, entity resolution, multigraph traversal, RRF reranking, and citation-grounded answer generation.
- **Local & Cost-Free**: Runs 100% locally with zero external API fees using **Ollama**:
  - **LLM**: `gemma3:4b` (for extraction, planning, summarization, and grounded generation)
  - **Embeddings**: `embeddinggemma:latest` (768-dimensional dense embeddings)
- **Directed Multigraph Knowledge Graphs**: Built on `igraph` with multigraph preservation (distinct relation types preserved, duplicates aggregated with merged chunk provenance).
- **Leiden Community Detection**: Unmutated undirected graph projection weighted by distinct supporting chunks (`RBConfigurationVertexPartition`, `resolution=1.0`, `seed=42`) with LLM-generated community reports.
- **Strict Multi-Domain Isolation**: Zero cross-knowledge-base leakage between multiple isolated domains (e.g., `academic_kb` vs `company_kb`).
- **Mathematical Hybrid Fusion (RRF)**: Reciprocal Rank Fusion ($k=60$) with deterministic tie-breaking prioritizing graph-grounded evidence.
- **Strict Factual Grounding & Abstention**: Context window character budgeting ($12,000$ chars) with mandatory inline citations `[chunk_id, doc_id, p.X]` and automatic abstention for unanswerable queries.

---

## Architecture Overview

```
Raw Documents (.pdf, .txt, .md)
       │
       ▼
[1] PDFLoader & Normalization
       │
       ▼
[2] DeterministicChunker (1100 chars, 180 overlap)
       │
       ▼
[3] Ontology-Bound Extraction (gemma3:4b) ──► Entity Mentions & Triples
       │
       ▼
[4] Conservative Entity Resolution (5-step) ──► Canonical Entities & Remapped Graph
       │
       ├──► [5] Vector Store (ChromaDB: entity_descriptions, text_chunks)
       │
       └──► [6] Directed Multigraph (igraph)
                  │
                  ├──► [7] Graph Validator (manifest.json, SHA-256 fingerprint)
                  │
                  └──► [8] Leiden Community Detection (seed=42, res=1.0)
                             │
                             └──► [9] Community Summarization ──► ChromaDB (community_reports)
```

Online Query Processing:
```
User Question
     │
     ▼
[10] Query Planner (Intent: LOCAL | GLOBAL | MIXED)
     │
     ├──► [11] Seed Entity Retriever (4-step hierarchy: exact -> alias -> vector -> tie-check)
     │          │
     │          ▼
     │    [12] Directed Graph Traverser (BFS, cycle-free, max_hops=3, scope filter)
     │
     ├──► [13] Dense Vector Retriever (ChromaDB text_chunks)
     │
     └──► [14] Community Retriever (ChromaDB community_reports)
                 │
                 ▼
          [15] Evidence Merger (deduplication by chunk_id)
                 │
                 ▼
          [16] Reciprocal Rank Fusion (RRF score: 1/(60+graph_rank) + 1/(60+vector_rank))
                 │
                 ▼
          [17] Context Builder (budget: 12,000 chars)
                 │
                 ▼
          [18] Grounded Answer Generator (inline citations [C001, D001, p.1] + abstention)
```

---

## Installation & Setup

### 1. Prerequisites
- **Python**: 3.10+ (tested on Python 3.13.7)
- **Ollama**: [ollama.com](https://ollama.com/)

Pull the required local models:
```bash
ollama pull gemma3:4b
ollama pull embeddinggemma:latest
```

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/shikhar-srivastava/hybrid-graphrag-system.git
cd hybrid-graphrag-system

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## CLI Usage

The system exposes a unified CLI entrypoint via `main.py`:

### Indexing a Knowledge Base
```bash
./.venv/bin/python main.py \
  --config configs/domains/academic.yaml \
  --index-dir /path/to/raw_documents
```

### Querying
```bash
# Full Hybrid Retrieval (Default)
./.venv/bin/python main.py \
  --config configs/domains/academic.yaml \
  --query "What technology does the project supervised by Rahul's mentor use?" \
  --mode hybrid

# Vector Search Only
./.venv/bin/python main.py \
  --config configs/domains/academic.yaml \
  --query "machine learning healthcare" \
  --mode vector_only

# Knowledge Graph Traversal Only
./.venv/bin/python main.py \
  --config configs/domains/academic.yaml \
  --query "Who works under Professor Sharma?" \
  --mode graph_only

# Community-Aware Macro Thematic Retrieval
./.venv/bin/python main.py \
  --config configs/domains/academic.yaml \
  --query "Summarize the primary research themes across all projects." \
  --mode community_aware
```

---

## Running the Test Suite

Run the complete deterministic test suite:
```bash
./.venv/bin/pytest -v
```

All 40 unit and integration tests run offline without external API dependencies in less than 1 second.

---

## License

Apache 2.0 License.
