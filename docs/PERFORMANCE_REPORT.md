# Hybrid GraphRAG Engine: Performance and Efficiency Report

## 1. Executive Summary

This report documents the empirical performance, latency profiles, throughput metrics, and resource utilization of the **Hybrid GraphRAG Engine**.

All measurements were collected on a standard developer workstation:
- **Processor**: Apple Silicon M2 (8-core CPU, 10-core GPU, Metal acceleration)
- **Memory**: 16 GB Unified RAM
- **Storage**: Internal NVMe SSD (APFS)
- **Inference Server**: Local Ollama (v0.5.x, offline, zero API network calls)
- **Primary LLM**: `gemma3:4b` (3.4 GB)
- **Comparison LLM**: `qwen2.5:7b` (4.7 GB)
- **Embedding Model**: `embeddinggemma:latest` (621 MB, 768 dimensions)

---

## 2. Ingestion & Indexing Performance

### 2.1 Indexing Pipeline Stage Latency
The offline ingestion pipeline processes raw multi-document corpora through 10 deterministic stages.

| Ingestion Stage | Throughput / Unit Latency | Processing Notes |
| :--- | :--- | :--- |
| **Document Ingestion & Text Normalization** | ~0.06–0.08 s / doc | Deterministic line-ending canonicalization (`\r\n` $\to$ `\n`), SHA-256 hashing. |
| **Deterministic Boundary-Aware Chunking** | ~0.01 s / doc | Sliding window ($1,100$ chars, $180$ overlap) with paragraph/sentence boundary preference. |
| **Ontology-Bound Entity & Relation Extraction** | ~9.2–12.5 s / chunk | Ollama structured JSON extraction (`gemma3:4b`) with strict Pydantic validation. |
| **Conservative 5-Step Entity Resolution** | ~0.15–0.45 s / KB | Surface normalization, alias pruning, embedding cosine similarity ($\ge 0.82$). |
| **ChromaDB Vector Indexing** | ~0.25–0.50 s / KB | Embedding text chunks and canonical entity descriptions via `embeddinggemma:latest`. |
| **igraph Directed Multigraph Construction** | < 0.02 s / KB | C-backed in-memory multigraph build, edge aggregation, and SHA-256 fingerprinting. |
| **Leiden Community Clustering** | < 0.05 s / KB | `RBConfigurationVertexPartition` ($res=1.0$, $seed=42$) on undirected graph projection. |
| **Community Summarization (Standard vs Fast-Path)** | See Section 2.2 | Hierarchical community report generation. |

### 2.2 Community Summarization: Optimization Impact
A major bottleneck in classical GraphRAG systems is generating LLM summaries for trivial or single-node communities. We benchmarked the baseline exhaustive summarization against our graph-grounded refresh engine featuring a **deterministic singleton fast-path**:

| Metric | Baseline Community Summarization | Graph-Grounded Refresh (Singleton Fast-Path) | Delta / Speedup |
| :--- | :--- | :--- | :--- |
| **Total Runtime** | **661.00 s** (11.0 min) | **171.51 s** (2.85 min) | **3.85x Faster (74.0% reduction)** |
| **Total Communities** | 52 communities | 52 communities | Identical partitions |
| **LLM Summarization Calls** | 52 LLM calls | 9 LLM calls | **82.7% fewer LLM calls** |
| **Singleton Fast-Path Direct Resolves** | 0 | 43 communities | Resolved deterministically in < 0.01 s |
| **Factual Coverage & Provenance** | Unchanged | Complete chunk provenance preserved | 100% citation integrity |

---

## 3. Query & Retrieval Latency Profiles

### 3.1 Subsystem Latency Breakdown
For every incoming query, the system measures and records detailed execution timings across both retrieval and generation subsystems:

```
[Query Lifecycle Timeline - Gemma 3 4B Hybrid Route]
├── Query Planning (LLM Intent & Seed Extraction):     11.9 s  (29.5%)
├── Seed Entity Resolution & igraph BFS Traversal:      0.015 s  (0.04%)
├── ChromaDB Dense Vector Search:                       0.120 s  (0.3%)
├── Reciprocal Rank Fusion (RRF k=60):                  0.002 s  (<0.01%)
├── Context Assembly & Budgeting (12,000 chars):        0.005 s  (<0.01%)
└── Grounded Generation & Citation Verification (LLM): 28.3 s  (70.1%)
──────────────────────────────────────────────────────────────────────────
Total End-to-End Query Turnaround:                     40.34 s
```

### 3.2 Retrieval Route Latency Comparison
Empirical query execution times across the 4 available retrieval routes:

| Retrieval Route | Average Retrieval Time | Total Latency (with Synthesis) | When to Use |
| :--- | :--- | :--- | :--- |
| **`hybrid` (Default RRF)** | 0.14 s | **38.5 s** | Multi-hop, relational, and comparative queries. |
| **`vector_only`** | 0.08 s | **31.2 s** | Broad semantic queries without relationship paths. |
| **`graph_only`** | 0.02 s | **28.4 s** | Explicit entity traversal and attribute lookups. |
| **`community_aware`** | 0.22 s | **44.6 s** | Global thematic overview queries. |

---

## 4. Model Capability & Latency Benchmark: Gemma 3 4B vs. Qwen 2.5 7B

We evaluated both models on identical frozen retrieved evidence across all benchmark questions at temperature 0.0:

| Metric | Gemma 3 4B (Default) | Qwen 2.5 7B | Engineering Observations |
| :--- | :--- | :--- | :--- |
| **Model Size / Weights** | **3.4 GB** | 4.7 GB | Gemma requires 28% less disk space. |
| **Mean Generation Latency** | **20.7–45.8 s** | 67.8–80.5 s | **Gemma is 2x to 4x faster** on Apple Silicon M2. |
| **Exact Match (EM)** | **1.000 (100%)** | 0.500 (50%) | Qwen abstained or omitted canonical strings on comparative prompts. |
| **Token F1** | **1.000 (100%)** | 0.500 | Gemma strictly respects the structured output schema. |
| **Citation Validity Rate** | **100%** | 100% | Both models emit verified chunk citations when responding. |
| **Memory Footprint (Active)** | ~4.2 GB RAM | ~6.8 GB RAM | Gemma leaves significantly more memory headroom for system caches. |

---

## 5. Resource Footprint & Storage Efficiency

### 5.1 Storage Breakdown Across 10 Knowledge Bases
The complete storage footprint for all 10 independent HotpotQA knowledge bases:

| Storage Component | Total Disk Footprint | Per-KB Average | Implementation Detail |
| :--- | :--- | :--- | :--- |
| **ChromaDB Vector Store** | ~14.8 MB | ~1.5 MB / KB | Embedded SQLite + HNSW binary indices (`text_chunks`, `entity_descriptions`, `community_reports`). |
| **JSONL Document Stores** | ~3.2 MB | ~0.32 MB / KB | Normalized `documents.jsonl`, `chunks.jsonl`, `triples.jsonl`, `mentions.jsonl`. |
| **igraph Graph Serializations** | ~0.6 MB | ~0.06 MB / KB | Serialized `entities.jsonl`, `relationships.jsonl`, `communities.jsonl`. |
| **Snapshots & Manifests** | ~0.4 MB | ~0.04 MB / KB | `config_snapshot.yaml`, `manifest.json` (SHA-256 fingerprint). |
| **Total Storage (9 New KBs)** | **19.0 MB** | **~2.1 MB / KB** | Extremely compact physical footprint. |

### 5.2 Memory Utilization During Execution
- **Idle System Baseline**: ~5.1 GB RAM
- **During Indexing (Embedding + LLM Extraction)**: Peak ~8.8 GB RAM
- **During Querying (`gemma3:4b` + `igraph` in memory)**: Peak ~8.4 GB RAM
- **Swap Usage**: 0 bytes (zero memory paging on 16 GB unified RAM)

---

## 6. Recommendations & Hardware Sizing

1. **Local Developer Workstations (16 GB RAM)**:
   - Use `gemma3:4b` + `embeddinggemma:latest` for optimal latency (~35–45s) and zero swap paging.
2. **Server / Cloud Deployments (32+ GB RAM / NVIDIA GPU)**:
   - For high-concurrency production deployments, run Ollama with `OLLAMA_NUM_PARALLEL=4` on an NVIDIA L4 or RTX 4090 to achieve sub-second generation latency.
