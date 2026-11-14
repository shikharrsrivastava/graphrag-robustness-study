# Enterprise Hybrid GraphRAG Engine: System Architecture

## 1. Executive Summary & Design Principles

The **Hybrid GraphRAG Engine** is an enterprise-grade, offline-first Retrieval-Augmented Generation system. It synthesizes the deterministic topological precision of directed property knowledge graphs with the semantic versatility of dense vector retrieval and hierarchical community clustering.

### Core Architectural Axioms
1. **Zero External API Dependencies**: Built 100% on local model infrastructure using **Ollama** (`gemma3:4b` / `qwen2.5:7b` for reasoning and extraction; `embeddinggemma:latest` for 768-dimensional dense representations).
2. **Deterministic Topological Grounding**: Facts, relations, and communities maintain immutable bi-directional provenance linkages back to original document text chunks and character spans.
3. **Multi-Domain Tenant Isolation**: Physical separation of storage, vector indices, graph partitions, and configuration snapshots per Knowledge Base ID (`kb_id`), preventing cross-domain information leakage.
4. **Deterministic Reasoning Engine**: Comparative queries (Boolean equivalence and entity selection) and multi-hop bridge questions are resolved through rule-grounded, verifiable verification layers rather than speculative generative hallucinations.

---

## 2. Ingestion Pipeline Architecture

The ingestion pipeline converts unstructured heterogeneous documents (PDFs, Markdown, text) into validated multigraph representations, vector indices, and hierarchical Leiden community summaries.

```mermaid
flowchart TD
    subgraph Ingestion ["1. Document Processing & Ingestion"]
        A["Raw Documents (.pdf, .txt, .md)"] --> B["PDFLoader & Normalizer"]
        B --> C["DeterministicChunker<br/>(1100 chars, 180 overlap)"]
    end

    subgraph Extraction ["2. Ontology-Bound Extraction"]
        C --> D["Ollama LLM (gemma3:4b)<br/>Structured Extraction Schema"]
        D --> E{"Ontology Validator"}
        E -- "Valid Triple" --> F["Entity Mentions & Triples"]
        E -- "Invalid/Negated" --> G["Rejection Log (manifest.json)"]
    end

    subgraph Resolution ["3. Conservative Entity Resolution"]
        F --> H["Surface Normalization"]
        H --> I["Honorific & Alias Stripping"]
        I --> J["Dense Vector Similarity (EmbeddingGemma)"]
        J --> K["Disambiguation & Remapping"]
        K --> L["Canonical Entities"]
    end

    subgraph Storage_Build ["4. Graph & Vector Indexing"]
        L --> M["igraph Directed Multigraph Builder"]
        C & L --> N["ChromaDB Collections<br/>(text_chunks, entity_descriptions)"]
        M --> O["Leiden Clustering (seed=42, res=1.0)"]
        O --> P["Community Detection (RBConfiguration)"]
        P --> Q["Graph-Grounded Summarizer"]
        Q --> R["ChromaDB (community_reports)"]
        M --> S["Graph Fingerprint (SHA-256) & Manifest"]
    end
```

### Ingestion Stages
1. **Document Loading & Normalization**: Deterministic alphabetical file ordering, sequential identifier assignment (`D001`, `D002`), line-ending canonicalization (`\r\n` $\to$ `\n`), per-page tracking.
2. **Deterministic Chunking**: Sliding-window chunker with character limit ($1,100$) and overlap ($180$). Boundary preference heuristics prioritize paragraphs, sentences, and punctuation within the final 25% boundary window.
3. **Ontology-Bound Extraction**: Extraction prompt restricted by strict domain ontology (`DomainOntology`). Rejects quote mismatches, negated facts, and invalid endpoint entity types.
4. **Conservative Entity Resolution**: 5-step resolution pipeline: surface matching $\to$ honorific stripping $\to$ exact/alias matching $\to$ embedding cosine similarity ($\ge 0.82$) $\to$ ambiguous band verification.
5. **Directed Multigraph Construction**: Preserves distinct relationship types on directed multigraph edges in `igraph`, aggregating duplicate edges and merging provenance chunk sets.
6. **Leiden Community Detection**: Projects unmutated undirected graph with distinct chunk support weights, computing Leiden partitions with fixed random seed ($42$).
7. **Graph-Grounded Community Summarization**: Hierarchical community report generation featuring deterministic fast-path for singletons and grounded LLM synthesis for multi-node clusters.

---

## 3. Retrieval & Fusion Pipeline

The retrieval pipeline dynamically adapts to query complexity, retrieving structural paths, semantic chunks, and high-level community overviews, combining them via Reciprocal Rank Fusion (RRF).

```mermaid
flowchart TD
    subgraph Query_Planning ["1. Query Analysis & Planning"]
        Q["User Query"] --> QP["Query Planner (LLM)"]
        QP --> Intent["Intent Classification<br/>(LOCAL, GLOBAL, MIXED)"]
        QP --> Seeds["Seed Entity Extraction"]
        QP --> Path["Planned Relationship Path"]
    end

    subgraph Retrieval_Streams ["2. Parallel Retrieval Streams"]
        Seeds --> SR["Seed Entity Resolver<br/>(Exact -> Alias -> Vector)"]
        SR --> GT["igraph BFS Traversal<br/>(Cycle-free, max_hops=3)"]
        Q --> VR["Dense Vector Search<br/>(ChromaDB text_chunks)"]
        Intent --> CR["Community Search<br/>(ChromaDB community_reports)"]
    end

    subgraph Fusion ["3. Evidence Merging & Rank Fusion"]
        GT --> EM["Evidence Deduplicator & Merger"]
        VR --> EM
        CR --> EM
        EM --> RRF["Reciprocal Rank Fusion (k=60)<br/>RRF = 1/(60+r_graph) + 1/(60+r_vector)"]
        RRF --> TieBreak["Deterministic Tie-Breaking<br/>(Supporting Edges, Provenance)"]
    end

    subgraph Generation ["4. Context Budgeting & Generation"]
        TieBreak --> CB["Context Builder<br/>(Budget: 12,000 characters)"]
        CB --> AG["Grounded Answer Generator"]
        AG --> Output["Structured QueryResult<br/>(Answer, Canonical Answer, Citations)"]
    end
```

### Retrieval Stages
1. **Query Planning**: LLM classifies question intent into `LOCAL` (entity-centric), `GLOBAL` (thematic overview), or `MIXED` (multi-hop relational). Identifies seed entities and predicted traversal paths.
2. **Seed Entity Resolution**: Matches seed entity specifications against canonical graph nodes using exact string, normalized alias, and embedding fallback.
3. **Directed Graph Traversal**: Breadth-First Search (BFS) up to `max_hops=3`, enforcing cycle prevention, optional tenant/scope filtering, and scoring edge type relevance.
4. **Dense Vector Retrieval**: HNSW cosine similarity lookup over text chunk embeddings.
5. **Hierarchical Community Retrieval**: Semantic discovery across macro-thematic community reports for global context.
6. **Reciprocal Rank Fusion (RRF)**: Merges candidates across retrieval channels with formula:
   $$\text{RRF}(d) = \sum_{m \in \{\text{graph}, \text{vector}\}} \frac{1}{k + r_m(d)} \quad (k=60)$$
   Ties are broken deterministically by graph supporting edge count, graph presence, and chunk ID.
7. **Context Construction**: Formats Community Overviews, Directed Graph Paths, and Text Chunks within a strict $12,000$ character window budget.

---

## 4. Deterministic Comparative & Multi-Hop Reasoning

To eliminate LLM hallucinations on comparative and multi-hop queries, the engine implements a deterministic verification layer.

```mermaid
flowchart TD
    Start["Incoming Query & Retrieved Evidence"] --> IntentCheck{"Query Intent Classification"}

    IntentCheck -- "Entity Selection<br/>('Which of A or B...')" --> SelBranch["ComparativeReasoningEngine.evaluate_selection"]
    IntentCheck -- "Boolean Comparison<br/>('Are A and B both...')" --> CompBranch["ComparativeReasoningEngine.evaluate_comparison"]
    IntentCheck -- "Multi-Hop Bridge<br/>('X of Y that was formed by who?')" --> BridgeBranch["Multi-Hop Traversal & Fact Extraction"]
    IntentCheck -- "General Factual" --> GenBranch["Direct Grounded Synthesis"]

    subgraph Selection_Reasoning ["Selection Reasoning"]
        SelBranch --> ExtractSeeds["Extract Verified Seed Attributes"]
        ExtractSeeds --> CheckCriteria{"Does Entity Satisfy Criteria?"}
        CheckCriteria -- "Entity 1 Only" --> Pick1["Select Entity 1"]
        CheckCriteria -- "Entity 2 Only" --> Pick2["Select Entity 2"]
        CheckCriteria -- "Both or Neither" --> AbstainSel["Abstain with Reason"]
        Pick1 & Pick2 --> SuffixCheck{"Post-Nominal Suffix in Evidence?<br/>(e.g. 'DSC', 'Jr.', 'OBE')"}
        SuffixCheck -- "Yes" --> AppendSuffix["Append Grounded Suffix"]
        SuffixCheck -- "No" --> RetainName["Retain Canonical Name"]
    end

    subgraph Comparison_Reasoning ["Boolean Comparison Reasoning"]
        CompBranch --> CompType{"Comparison Domain"}
        CompType -- "Occupation" --> SpecCheck{"Evidence-Grounded Specialization<br/>(e.g. 'opera composer')"}
        CompType -- "Location" --> GranCheck{"Location Granularity<br/>(neighborhood vs city vs country)"}
        CompType -- "General/Nationality" --> NormCheck{"Demonym & Synonym Normalization"}
        SpecCheck & GranCheck & NormCheck --> EquivResult{"Are Attributes Equivalent?"}
        EquivResult -- "Identical" --> AnsYes["canonical_answer: 'yes'"]
        EquivResult -- "Different" --> AnsNo["canonical_answer: 'no'"]
        EquivResult -- "Insufficient" --> AnsAbstain["Abstain"]
    end

    subgraph Bridge_Reasoning ["Multi-Hop Bridge Reasoning"]
        BridgeBranch --> Step1["Hop 1: Resolve Premise to Bridge Entity<br/>(e.g. '2014 S/S' -> 'WINNER')"]
        Step1 --> Disambig{"Distractor Check<br/>(Match all relational criteria)"}
        Disambig --> Step2["Hop 2: Resolve Bridge Entity to Target Attribute<br/>(e.g. 'WINNER' -> 'YG Entertainment')"]
        Step2 --> CanonTarget["canonical_answer = Target Attribute<br/>(Never intermediate entity)"]
    end
```

---

## 5. Storage Layer & Multi-Tenant Knowledge Base Isolation

All data artifacts are physically isolated within dedicated Knowledge Base directories:

```mermaid
graph TD
    Root["knowledge_bases/"]
    Root --> KB1["knowledge_bases/kb_finance/"]
    Root --> KB2["knowledge_bases/kb_legal/"]
    Root --> KB3["knowledge_bases/kb_hotpotqa/"]

    subgraph Layout ["Isolated Knowledge Base Directory Layout"]
        KB3 --> Snapshot["config_snapshot.yaml (Active Configuration)"]
        KB3 --> Manifest["manifest.json (Graph SHA-256 & Verification)"]
        KB3 --> DataDir["data/"]
        KB3 --> GraphDir["graph/"]
        KB3 --> VectorDir["vector_store/ (ChromaDB Directory)"]

        DataDir --> Chunks["processed/chunks.jsonl"]
        DataDir --> Mentions["processed/entity_mentions.jsonl"]
        DataDir --> Triples["processed/triples.jsonl"]

        GraphDir --> Entities["graph/entities.jsonl"]
        GraphDir --> Relations["graph/relationships.jsonl"]
        GraphDir --> Comm["graph/communities.jsonl"]
        GraphDir --> Reports["graph/community_reports.jsonl"]

        VectorDir --> ColChunks["Collection: text_chunks (Cosine HNSW)"]
        VectorDir --> ColEntities["Collection: entity_descriptions (Cosine HNSW)"]
        VectorDir --> ColReports["Collection: community_reports (Cosine HNSW)"]
    end
```

### Isolation Guarantees
- **Config Immobility**: Once initialized or indexed, the configuration snapshot (`config_snapshot.yaml`) locks the chunking parameters, ontology, and embedding model.
- **Data Encapsulation**: ChromaDB collections and igraph indices exist strictly inside the `kb_id` subdirectory; queries to `kb_finance` cannot access vectors or vertices from `kb_legal`.
- **Integrity Verification**: `manifest.json` preserves the SHA-256 fingerprint of the serialized multigraph, ensuring corruption or accidental file modification is detected on load.
