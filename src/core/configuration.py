"""Configuration management, merging, validation, and deterministic hashing."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from src.core.ontology import DomainOntology


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_size: int = Field(default=1100, gt=50)
    chunk_overlap: int = Field(default=180, ge=0)
    respect_sentence_boundaries: bool = True

    @model_validator(mode="after")
    def check_overlap_less_than_size(self) -> "ChunkingConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than chunk_size ({self.chunk_size})"
            )
        return self


class OllamaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(default="http://localhost:11434")
    llm_model: str = Field(default="gemma3:4b")
    embedding_model: str = Field(default="embeddinggemma:latest")
    expected_embedding_dim: int = Field(default=768, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_seconds: int = Field(default=120, gt=0)
    max_retries: int = Field(default=2, ge=0)


class ResolutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    similarity_candidate_threshold: float = Field(default=0.82, ge=0.0, le=1.0)
    auto_merge_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    require_type_match: bool = True
    use_llm_verification_for_ambiguous: bool = True


class LeidenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: str = Field(default="RBConfigurationVertexPartition")
    resolution_parameter: float = Field(default=1.0, gt=0.0)
    random_seed: int = Field(default=42)
    edge_weight_rule: str = Field(default="distinct_supporting_chunks")


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_traversal_hops: int = Field(default=3, ge=1, le=6)
    max_graph_candidates: int = Field(default=20, ge=1)
    vector_top_k: int = Field(default=10, ge=1)
    community_top_k: int = Field(default=3, ge=1)
    final_context_chunks: int = Field(default=8, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    max_context_chars: int = Field(default=12000, ge=500)
    provenance_expansion_enabled: bool = Field(
        default=False,
        description="Enable one-hop Entity-Chunk-Entity provenance expansion in graph retrieval",
    )
    max_chunks_per_cooccurring_entity: int = Field(
        default=2,
        ge=1,
        description="Hub suppression threshold: maximum chunks an entity can appear in before being excluded as a hub",
    )
    max_cooccurring_entities_per_chunk: int = Field(
        default=5,
        ge=1,
        description="Maximum co-occurring entities to expand per direct seed chunk",
    )
    max_expanded_candidates_per_query: int = Field(
        default=10,
        ge=1,
        description="Maximum total expanded chunk candidates per query",
    )
    direct_seed_outranks_expanded: bool = Field(
        default=True,
        description="Whether direct seed chunks always outrank bipartite-expanded chunks in graph candidate ranking",
    )
    max_candidates_per_seed: int = Field(
        default=5,
        ge=1,
        description="Maximum ambiguous canonical entity candidates to retain per seed specification in confidence-aware resolution",
    )
    document_title_discovery_enabled: bool = Field(
        default=False,
        description="Enable document-metadata-based entity discovery in graph retrieval",
    )
    max_title_matches_per_seed: int = Field(
        default=2,
        ge=1,
        description="Maximum document titles to match per seed specification",
    )
    max_chunks_per_title_match: int = Field(
        default=3,
        ge=1,
        description="Maximum chunks to retrieve per matched document title",
    )
    max_title_candidates_per_query: int = Field(
        default=6,
        ge=1,
        description="Maximum total chunks to admit via document-title discovery per query",
    )
    entity_description_discovery_enabled: bool = Field(
        default=False,
        description="Enable entity-description-based seed discovery in graph retrieval",
    )
    entity_description_similarity_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity threshold for entity description matching",
    )
    max_description_candidates_per_seed: int = Field(
        default=3,
        ge=1,
        description="Maximum entity description candidates to retain per seed specification",
    )
    max_description_candidates_per_query: int = Field(
        default=6,
        ge=1,
        description="Maximum total entity description candidates to admit per query",
    )


class PromptVersions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_extraction: str = Field(default="v1.0")
    relationship_extraction: str = Field(default="v1.0")
    entity_resolution: str = Field(default="v1.0")
    community_summary: str = Field(default="v1.0")
    query_planning: str = Field(default="v1.0")
    answer_generation: str = Field(default="v1.0")


class EngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kb_id: str = Field(..., min_length=1)
    config_version: str = Field(default="1.0.0")
    storage_root: str = Field(default="knowledge_bases")
    prompts_dir: str = Field(default="prompts")
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    resolution: ResolutionConfig = Field(default_factory=ResolutionConfig)
    leiden: LeidenConfig = Field(default_factory=LeidenConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    prompt_versions: PromptVersions = Field(default_factory=PromptVersions)
    ontology: DomainOntology

    def compute_config_hash(self) -> str:
        """Compute deterministic SHA-256 hash of the entire active configuration."""
        canonical_json = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def get_kb_root(self, base_dir: Path = Path(".")) -> Path:
        return base_dir / self.storage_root / self.kb_id

    def initialize_kb_directories(self, base_dir: Path = Path(".")) -> Dict[str, Path]:
        """Create isolated directory tree for this knowledge base and save config_snapshot.yaml."""
        kb_root = self.get_kb_root(base_dir)
        paths = {
            "kb_root": kb_root,
            "raw_documents": kb_root / "data" / "raw_documents",
            "processed": kb_root / "data" / "processed",
            "graph": kb_root / "graph",
            "vector_store": kb_root / "vector_store",
        }
        for p in paths.values():
            p.mkdir(parents=True, exist_ok=True)

        snapshot_path = kb_root / "config_snapshot.yaml"
        if not snapshot_path.exists():
            with open(snapshot_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=False)

        return paths


def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override dictionary into base dictionary without mutating inputs."""
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = deep_merge_dicts(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_engine_config(
    domain_config_path: Path,
    default_config_path: Path = Path("configs/default.yaml"),
    kb_id_override: Optional[str] = None,
) -> EngineConfig:
    """Load default YAML config, merge domain YAML overrides, and validate with Pydantic."""
    base_data: Dict[str, Any] = {}
    if default_config_path.exists():
        with open(default_config_path, "r", encoding="utf-8") as f:
            base_data = yaml.safe_load(f) or {}

    with open(domain_config_path, "r", encoding="utf-8") as f:
        domain_data = yaml.safe_load(f) or {}

    merged = deep_merge_dicts(base_data, domain_data)
    if kb_id_override:
        merged["kb_id"] = kb_id_override

    return EngineConfig.model_validate(merged)
