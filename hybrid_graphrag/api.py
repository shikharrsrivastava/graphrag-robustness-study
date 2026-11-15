"""Public Python API facade for the Hybrid GraphRAG system."""

from __future__ import annotations
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
import yaml

from src.core.configuration import EngineConfig, deep_merge_dicts, load_engine_config
from src.core.interfaces import EmbeddingProvider
from src.core.models import RetrievalResult
from src.engine import HybridGraphRAGEngine
from src.extraction.schemas import OllamaLLMClient
from hybrid_graphrag.exceptions import (
    ConfigurationError,
    IngestionError,
    KnowledgeBaseNotFoundError,
    QueryError,
)
from hybrid_graphrag.models import Citation, QueryResult

logger = logging.getLogger("hybrid_graphrag")


class HybridGraphRAG:
    """High-level public facade for the enterprise Hybrid GraphRAG engine.

    Provides a clean, domain-agnostic API for configuring knowledge bases,
    ingesting documents, executing hybrid retrieval queries, and accessing
    structured, evidence-grounded answers with citations.
    """

    def __init__(self, engine: HybridGraphRAGEngine) -> None:
        self._engine = engine
        self._closed = False

    @property
    def kb_id(self) -> str:
        """The active knowledge base identifier."""
        return self._engine.kb_id

    @property
    def kb_dir(self) -> Path:
        """The directory path of the active knowledge base."""
        if hasattr(self._engine, "kb_paths") and "kb_root" in self._engine.kb_paths:
            return Path(self._engine.kb_paths["kb_root"])
        return Path(self._engine.config.get_kb_root(getattr(self._engine, "base_dir", Path("."))))

    @property
    def config(self) -> EngineConfig:
        """The underlying active engine configuration."""
        return self._engine.config

    @classmethod
    def from_config(
        cls,
        config: Union[str, Path, Dict[str, Any], EngineConfig],
        kb_id: Optional[str] = None,
        base_dir: Union[str, Path] = ".",
        llm_client: Optional[OllamaLLMClient] = None,
        embedding_service: Optional[EmbeddingProvider] = None,
    ) -> "HybridGraphRAG":
        """Initialize HybridGraphRAG from a YAML file, dictionary, or EngineConfig object.

        Args:
            config: Path to a domain/snapshot YAML file, configuration dictionary, or EngineConfig.
            kb_id: Optional knowledge base identifier override.
            base_dir: Base directory for storage resolution (defaults to ".").
            llm_client: Optional custom LLM client (e.g. for testing or non-Ollama providers).
            embedding_service: Optional custom embedding provider.

        Returns:
            HybridGraphRAG facade instance.
        """
        engine_config: EngineConfig

        if isinstance(config, EngineConfig):
            engine_config = config
        elif isinstance(config, dict):
            data = dict(config)
            if kb_id:
                data["kb_id"] = kb_id
            if "ontology" not in data:
                base_cfg_path = Path(base_dir) / "configs" / "default.yaml"
                domain_cfg_path = Path(base_dir) / "configs" / "domains" / "academic.yaml"
                base_data: Dict[str, Any] = {}
                if base_cfg_path.exists():
                    with open(base_cfg_path, "r", encoding="utf-8") as f:
                        base_data = yaml.safe_load(f) or {}
                if domain_cfg_path.exists():
                    with open(domain_cfg_path, "r", encoding="utf-8") as f:
                        domain_data = yaml.safe_load(f) or {}
                    base_data = deep_merge_dicts(base_data, domain_data)
                data = deep_merge_dicts(base_data, data)
            try:
                engine_config = EngineConfig.model_validate(data)
            except Exception as e:
                raise ConfigurationError(f"Invalid configuration dictionary: {e}") from e
        elif isinstance(config, (str, Path)):
            cfg_path = Path(config)
            if not cfg_path.exists():
                raise ConfigurationError(f"Configuration file not found: {cfg_path}")

            try:
                # 1. Try loading as a domain config merged with default.yaml
                if "domains" in str(cfg_path) or cfg_path.name in ("academic.yaml", "legal.yaml", "company.yaml", "medical_research.yaml"):
                    engine_config = load_engine_config(
                        domain_config_path=cfg_path,
                        kb_id_override=kb_id,
                    )
                else:
                    # 2. Try loading as a standalone full configuration or snapshot
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                    if not isinstance(data, dict):
                        raise ConfigurationError(f"Configuration file must contain a YAML mapping: {cfg_path}")
                    if kb_id:
                        data["kb_id"] = kb_id
                    engine_config = EngineConfig.model_validate(data)
            except Exception as e:
                raise ConfigurationError(f"Failed to load configuration from {cfg_path}: {e}") from e
        else:
            raise ConfigurationError(f"Unsupported configuration type: {type(config)}")

        if kb_id:
            # Revalidate with overridden kb_id
            cfg_dict = engine_config.model_dump(mode="json")
            cfg_dict["kb_id"] = kb_id.strip()
            engine_config = EngineConfig.model_validate(cfg_dict)

        # Validate knowledge base identifier
        if not engine_config.kb_id or not engine_config.kb_id.strip():
            raise ConfigurationError("Knowledge base identifier (kb_id) must be a non-empty string.")
        if "/" in engine_config.kb_id or "\\" in engine_config.kb_id or "\0" in engine_config.kb_id:
            raise ConfigurationError(f"Invalid kb_id '{engine_config.kb_id}': path separators are not permitted.")

        try:
            engine = HybridGraphRAGEngine(
                config=engine_config,
                llm_client=llm_client,
                embedding_service=embedding_service,
                base_dir=base_dir,
            )
            return cls(engine)
        except Exception as e:
            raise ConfigurationError(f"Engine initialization failed: {e}") from e

    @classmethod
    def from_existing(
        cls,
        kb_dir: Union[str, Path],
        base_dir: Union[str, Path] = ".",
        llm_client: Optional[OllamaLLMClient] = None,
        embedding_service: Optional[EmbeddingProvider] = None,
    ) -> "HybridGraphRAG":
        """Reopen an existing, already-indexed knowledge base without re-ingesting documents.

        Args:
            kb_dir: Path to the knowledge base root directory (e.g. 'knowledge_bases/my_kb').
            base_dir: Base directory for relative storage paths.
            llm_client: Optional custom LLM client.
            embedding_service: Optional custom embedding provider.

        Returns:
            HybridGraphRAG facade instance attached to the existing KB.
        """
        target_path = Path(kb_dir)
        if not target_path.exists() or not target_path.is_dir():
            raise KnowledgeBaseNotFoundError(f"Knowledge base directory does not exist: {target_path}")

        snapshot_path = target_path / "config_snapshot.yaml"
        if not snapshot_path.exists():
            raise KnowledgeBaseNotFoundError(
                f"Knowledge base snapshot file not found in {target_path}. Cannot reopen uninitialized KB."
            )

        with open(snapshot_path, "r", encoding="utf-8") as f:
            cfg_data = yaml.safe_load(f)

        # Ensure storage_root and kb_id align with target directory
        cfg_data["kb_id"] = target_path.name
        existing_root = cfg_data.get("storage_root")
        if not existing_root or not (Path(base_dir) / existing_root / cfg_data["kb_id"]).exists():
            cfg_data["storage_root"] = str(target_path.parent)

        engine_config = EngineConfig.model_validate(cfg_data)
        engine = HybridGraphRAGEngine(
            config=engine_config,
            llm_client=llm_client,
            embedding_service=embedding_service,
            base_dir=base_dir,
        )
        # Pre-load graph from store
        engine.get_or_load_graph()
        return cls(engine)

    def ingest(self, input_path: Union[str, Path], rebuild: bool = True) -> Dict[str, Any]:
        """Ingest and index documents from a directory or PDF file.

        Args:
            input_path: Path to a directory containing PDF documents.
            rebuild: Whether to clear previous records and rebuild this KB (default: True).

        Returns:
            Dictionary containing indexing summary statistics.
        """
        self._check_closed()
        in_path = Path(input_path)
        if not in_path.exists():
            raise IngestionError(f"Document input path does not exist: {in_path}")

        try:
            return self._engine.index_directory(in_path, rebuild=rebuild)
        except Exception as e:
            raise IngestionError(f"Document ingestion failed: {e}") from e

    def query(
        self,
        question: str,
        mode: str = "hybrid",
        route: Optional[str] = None,
        scope_id: Optional[str] = None,
        comparative_reasoning_mode: Optional[str] = None,
    ) -> QueryResult:
        """Execute retrieval and answer generation for a user question.

        Args:
            question: The natural language question to answer.
            mode: Retrieval mode: 'hybrid' (default), 'vector_only', 'graph_only', or 'community_aware'.
            route: Alias for mode parameter for backward compatibility.
            scope_id: Optional tenant or document scope filter.
            comparative_reasoning_mode: Optional comparative reasoning mode ('enforce' or 'off').

        Returns:
            QueryResult containing the generated answer, canonical answer, citations, and metadata.
        """
        self._check_closed()
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if route is not None:
            mode = route

        allowed_modes = {"hybrid", "vector_only", "graph_only", "community_aware"}
        mode_clean = mode.strip().lower()
        if mode_clean not in allowed_modes:
            raise ValueError(
                f"Unknown retrieval mode: '{mode}'. Allowed modes: {', '.join(sorted(allowed_modes))}"
            )

        query_started = time.perf_counter()
        try:
            answer_text, retrieval_result = self._engine.query(
                question=question,
                mode=mode_clean,
                scope_id=scope_id,
                comparative_reasoning_mode=comparative_reasoning_mode,
            )
        except Exception as e:
            raise QueryError(f"Query execution failed: {e}") from e
        total_latency = time.perf_counter() - query_started

        # Construct structured citations
        evidence_by_id = {ev.chunk_id: ev for ev in retrieval_result.evidence_list}
        citations: List[Citation] = []

        for cid in retrieval_result.cited_chunk_ids:
            if cid in evidence_by_id:
                ev = evidence_by_id[cid]
                snippet = ev.text.strip()
                if len(snippet) > 150:
                    snippet = snippet[:150] + "..."
                citations.append(
                    Citation(
                        chunk_id=ev.chunk_id,
                        doc_id=ev.doc_id,
                        filename=ev.filename,
                        page_numbers=list(ev.page_numbers),
                        text_snippet=snippet,
                        valid=True,
                    )
                )

        timing_data: Dict[str, float] = {}
        if "query_planning_seconds" in retrieval_result.route_diagnostics:
            timing_data["query_planning_seconds"] = retrieval_result.route_diagnostics["query_planning_seconds"]
        if "answer_generation_seconds" in retrieval_result.route_diagnostics:
            timing_data["answer_generation_seconds"] = retrieval_result.route_diagnostics["answer_generation_seconds"]

        return QueryResult(
            question=question.strip(),
            answer=retrieval_result.generated_answer or answer_text,
            canonical_answer=retrieval_result.canonical_answer,
            abstained=retrieval_result.abstained,
            abstention_reason=retrieval_result.abstention_reason,
            citations=citations,
            cited_chunk_ids=list(retrieval_result.cited_chunk_ids),
            mode=mode_clean,
            latency_seconds=total_latency,
            timing=timing_data,
            raw_result=retrieval_result,
            generation_error=getattr(retrieval_result, "generation_error", False),
        )

    def get_kb_stats(self) -> Dict[str, Any]:
        """Return summary statistics of the active knowledge base."""
        self._check_closed()
        graph = self._engine.get_or_load_graph()
        return {
            "kb_id": self._engine.kb_id,
            "documents": len(self._engine.doc_store._doc_index),
            "chunks": len(self._engine.doc_store._chunk_index),
            "entities": len(self._engine.doc_store._canonical_entities),
            "relationships": len(self._engine.doc_store._graph_relationships),
            "communities": len(self._engine.doc_store._communities),
            "community_reports": len(self._engine.doc_store._community_reports),
            "vector_store_counts": {
                "chunks": self._engine.vector_store.chunk_collection.count(),
                "entities": self._engine.vector_store.entity_collection.count(),
                "community_reports": self._engine.vector_store.community_collection.count(),
            },
            "graph": {
                "vertices": graph.vcount(),
                "edges": graph.ecount(),
            },
        }

    def close(self) -> None:
        """Release in-memory graph caches and resource handles."""
        if not self._closed:
            self._engine._graph = None
            self._engine.doc_store._doc_index.clear()
            self._engine.doc_store._chunk_index.clear()
            self._closed = True

    def _check_closed(self) -> None:
        if self._closed:
            raise RuntimeError("HybridGraphRAG engine instance has already been closed.")

    def __enter__(self) -> "HybridGraphRAG":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
