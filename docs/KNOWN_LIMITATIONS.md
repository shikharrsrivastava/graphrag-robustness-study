# Known Limitations and Technical Assessment

This document provides a transparent, honest engineering assessment of the current design boundaries, performance trade-offs, and scaling limits of the **Hybrid GraphRAG Engine**.

---

## 1. Local Inference Latency vs. Cloud APIs

### Trade-off Overview
Running entirely on local, cost-free model infrastructure via **Ollama** ensures absolute data privacy, zero API bills, and offline air-gapped readiness. However, local inference carries measurable latency costs on consumer hardware.

### Empirical Latency Profiles (Apple Silicon M2, 16 GB Unified Memory)
| Model Size | Average Planning Latency | Average Generation Latency | Total Query Turnaround |
| :--- | :--- | :--- | :--- |
| **Gemma 3 4B (3.4 GB)** | 10–18 s | 20–35 s | 30–55 s |
| **Qwen 2.5 7B (4.7 GB)** | 25–40 s | 50–85 s | 75–125 s |
| *Cloud Frontier (e.g. Claude 3.5 Sonnet)* | *~1.5 s* | *~2.5 s* | *~4 s* |

### Mitigation & Guidance
- For high-throughput automated batch evaluation or latency-critical applications, local 4B parameters represent the Pareto-optimal frontier between reasoning precision and execution speed.
- The architecture is modular: cloud LLM endpoints (e.g., Anthropic, OpenAI, Google Gemini) can be plugged in by implementing the `OllamaLLMClient` protocol without modifying indexing or retrieval pipelines.

---

## 2. Document Scale and Single-Node In-Memory Graphs

### In-Memory Graph Representation (`igraph`)
The graph engine represents knowledge graphs in-memory using C-based `igraph.Graph` data structures.
- **Current Bound**: Tested up to ~50,000 entities and ~200,000 relationships per knowledge base on a 16 GB machine.
- **Limitation**: Very large corporate knowledge graphs exceeding millions of nodes cannot reside entirely within single-process RAM.
- **Future Direction**: Support distributed or disk-backed graph stores (e.g. Neo4j, Kùzu, or Memgraph) for enterprise-scale deployments exceeding $10^7$ edges.

### Vector Storage Scalability (`ChromaDB`)
- The embedded ChromaDB vector store uses local HNSW indices.
- While performant for corpora under 50,000 chunks, massive multi-tenant installations require dedicated vector database clusters (e.g., Qdrant, Milvus, or pgvector).

---

## 3. Community Detection Granularity (Leiden)

### Resolution Sensitivity
- The engine uses the Leiden community detection algorithm (`RBConfigurationVertexPartition` with `resolution=1.0` and fixed random `seed=42`).
- On small document corpora (e.g. 10 candidate context documents, ~15–30 nodes), Leiden frequently identifies isolated singletons or tiny two-node communities.
- **Implemented Optimization**: The engine employs a deterministic *singleton fast-path*, generating structured community summaries deterministically without invoking expensive LLM summarization calls for 1-node trivial communities.
- For massive corpora, multi-level hierarchical Leiden clustering (recursive sub-community decomposition) should be tuned according to the corpus density.

---

## 4. Multi-Hop Graph Traversal Depth

### Combinatorial Explosion
- Breadth-First Search (BFS) graph traversal is bounded by default to `max_hops=3`.
- In dense multigraphs with high vertex degree (e.g., central entities connected to hundreds of attributes), unconstrained traversal beyond 3 hops leads to exponential vertex expansion:
  $$\mathcal{O}(d^h) \quad \text{where } d = \text{average degree}, h = \text{hops}$$
- **Constraint**: Multi-hop reasoning chains requiring more than 3 distinct hops must either be partitioned into sequential sub-queries or guided by explicit relationship path constraints from the `QueryPlan`.

---

## 5. Entity Disambiguation and Homonyms

### Surface Form Ambiguity
- The conservative entity resolution pipeline relies on a combination of surface normalization, honorific stripping, and dense embedding cosine similarity (threshold $\ge 0.82$).
- **Limitation**: Entities sharing exact or near-identical surface names in distinct contexts (e.g. "Michael Jordan" the computer scientist vs. "Michael Jordan" the athlete) can be conflated if their source documents do not contain distinguishing contextual entity types.
- **Mitigation**: Domain ontologies (`DomainOntology`) enforce expected entity type constraints (e.g. `PERSON`, `ORGANIZATION`, `LOCATION`), preventing invalid cross-type merges.

---

## 6. Document Format Support

- The core pipeline supports native text, Markdown, and digital PDF documents via `pypdf`.
- **Limitation**: Scanned PDFs containing only bitmap images without an embedded text layer are flagged with `is_scanned=True` and skipped during chunking. Optical Character Recognition (OCR) must be run prior to ingestion.

---

## 7. Comparative Reasoning Mode and Benchmark Specialization

- The system configuration parameter `comparative_reasoning_mode` defaults to `"off"`.
- In `"off"` mode, the engine passes the LLM's grounded generation output directly through without enforcing deterministic post-processing overrides, making it safe for general enterprise domains and custom corpora.
- When explicitly set to `"enforce"`, the engine engages deterministic attribute evaluation heuristics (such as demonym normalization, location granularity checking between neighborhoods and cities, and occupational specializations such as 'opera composer'). These heuristics were specialized to achieve optimal accuracy on the HotpotQA development set (where 100% pilot accuracy was achieved).
- Because these rules contain benchmark-tuned regex patterns and assumptions (e.g. nationality/location categorizations tailored to HotpotQA dev-set entity pairs), `"off"` is the production default to prevent unexpected overrides on non-HotpotQA corpora.

---

## 8. Roadmap & Planned Improvements

1. **Pluggable Cloud LLM Adapters**: Native litellm / async client support for hybrid local/cloud routing.
2. **Dynamic Context Chunk Compaction**: Sub-chunk token compression to fit larger multi-hop evidence graphs into smaller context windows.
3. **Kùzu / Neo4j Graph Backend Integration**: Optional persistent disk-backed graph engine alongside in-memory igraph.
4. **Active Disambiguation Feedback Loop**: Interactive clarification prompts when seed entity resolution encounters tied candidate entities.

---

## 9. Synthetic-KG Audit Findings (added 2026-11-16)

Evidence: `experiments/synth_kg_fix/` (Phase 0 re-verification, Phase 2 dev evaluation on seed 7, Phase 3 fresh-seed run).

- **No coreference step.** Sentences whose subject is a pronoun ("They is an official organizational part of ...", "It is directly responsible for developing X") lose their relationship edge: the endpoint-in-evidence check rejects them (`REJECTED_ENDPOINT_NOT_IN_EVIDENCE`) or the model reverses the direction.
- **Partial-name aliases are not resolved.** Seed queries such as "Zantonnax Division" for "Zantonnax Infrastructure Group", or "Haelenrix" for "Haelenrix Gateway", resolve to nothing when several entities could share the first token. Only initial + surname forms of person-like entities were added (5 of the 15 seed-7 alias questions remained unresolved after the fixes).
- **Extraction errors create wrong KG edges that can reach answers.** The model sometimes types an employee as PRODUCT and emits `DEVELOPS` instead of `MANAGES`, or reverses an edge. Optional post-resolution re-validation removes such edges only when the canonical type contradicts the ontology; it also removes links that were useful for traversal.
- **New behaviour is config-controlled and off by default.** `resolution.merge_exact_name_across_types`, `resolution.revalidate_relationships_after_resolution`, `resolution.enable_initial_surname_person_merge` (all `false` in `configs/default.yaml`; enabled only in `configs/domains/company.yaml`) and `prompt_versions.relationship_extraction_version` (`v1.0` default; `v1.1` adds a multi-relation instruction and was not enabled because it did not pass its dev rule). The enabling decisions were made on dev data (seed 7).
- **`graph_context_mode`:** the boolean `false` still sends the legacy "Supporting Graph Traversal Paths" block, whose `GRAPH_EDGE_PROVENANCE: ...` marker text appears in generated answers (17 of 480 answers in the seed-7 runs). Use the string `"off"` (no graph block, community summaries kept) or `"none"` (no graph and no community text) to avoid it.
- **Person-name matching** accepts initial + surname for every ontology (`person_names_compatible`), so "J. Smith" now matches "John Smith" in seed retrieval for PERSON ontologies too.
- **Hardcoded literal remaining:** `"PROJECT"` in `src/resolution/entity_resolver.py` (fallback generic-descriptor list) is still a domain-specific entity-type literal.
- **Prompt-version labels:** the older metadata key `prompt_versions.relationship_extraction` (default "v1.1") does not select a file; the file is selected by `relationship_extraction_version`, whose "v1.0" is the file the old label calls "v1.1".
- **Seed-7 benchmark issues** (`experiments/synth_kg/`): 14 of the 15 alias strings in the alias questions occur in no document (they test query-time resolution only); 28 of 32 "managed_client" unanswerable evaluations are near-answerable because the corpus states "serves" (figure from the earlier audit, not re-verified in the 2026-11-16 re-verification); the original "validated path exists" flag is an undirected, untyped reachability test and was unreliable (it is satisfied through shared client or department hubs without any gold edge); the audit script `synth_kg_audit/followup/part_c_alias_audit.py` contains invented names for Q_0091 and Q_0094.
