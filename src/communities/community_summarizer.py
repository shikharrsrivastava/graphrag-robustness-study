"""Community summarization generating grounded CommunityReport objects and vector indexing."""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import igraph
from pydantic import BaseModel, ConfigDict, Field
from src.core.interfaces import EmbeddingProvider
from src.core.models import CommunityReport
from src.extraction.schemas import OllamaLLMClient
from src.storage.document_store import DocumentStore
from src.storage.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

DEFAULT_COMMUNITY_SUMMARY_PROMPT = """You are an expert intelligence analyst and knowledge synthesizer. Your task is to generate a comprehensive, grounded summary report for a community of related entities in a knowledge graph.

COMMUNITY ID: {community_id}

MEMBER ENTITIES:
{entities_json}

RELATIONSHIPS:
{relationships_json}

SUPPORTING EVIDENCE & TEXT CHUNKS:
{supporting_chunks_text}

INSTRUCTIONS:
1. Synthesize the core theme, mission, or shared context uniting the entities in this community.
2. Ground all claims strictly in the provided member entities, relationships, and supporting text chunks. Do not hallucinate external facts.
3. Highlight key relationships, responsibilities, or interactions between members.
4. Return ONLY a valid JSON object matching this schema:

{
  "summary_text": "Detailed factual synthesis of this community's theme, members, and interactions."
}
"""


class CommunitySummaryOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    summary_text: str = Field(..., min_length=1)


class CommunitySummarizer:
    """Generates grounded CommunityReport summaries using an LLM and indexes them into ChromaDB."""

    def __init__(
        self,
        llm_client: OllamaLLMClient,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        doc_store: Optional[DocumentStore] = None,
        prompt_template_path: Optional[Union[str, Path]] = "prompts/community_summary.txt",
        prompt_template: Optional[str] = None,
        model_tag: Optional[str] = None,
        prompt_version: str = "v1.0",
    ) -> None:
        self.llm_client = llm_client
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.doc_store = doc_store
        self.model_tag = model_tag or self.llm_client.config.llm_model
        self.prompt_version = prompt_version

        if prompt_template is not None:
            self.prompt_template = prompt_template
        elif prompt_template_path is not None and Path(prompt_template_path).exists():
            self.prompt_template = Path(prompt_template_path).read_text(encoding="utf-8")
        else:
            self.prompt_template = DEFAULT_COMMUNITY_SUMMARY_PROMPT

    def summarize_communities(
        self,
        communities: List[Dict[str, Any]],
        graph: igraph.Graph,
        doc_store: Optional[DocumentStore] = None,
    ) -> List[CommunityReport]:
        """Generate CommunityReport summaries for each community and optionally index them in ChromaDB."""
        active_doc_store = doc_store or self.doc_store
        reports: List[CommunityReport] = []

        for comm in communities:
            comm_id = comm["community_id"]
            member_ids: List[str] = comm["member_entity_ids"]
            member_set = set(member_ids)

            # 1. Gather member entities from graph vertices
            member_vertices = [v for v in graph.vs if v["entity_id"] in member_set]
            entities_data = [
                {
                    "entity_id": v["entity_id"],
                    "name": v.attributes().get("label", ""),
                    "entity_type": v.attributes().get("entity_type", ""),
                    "description": v.attributes().get("description", ""),
                    "aliases": v.attributes().get("aliases", []),
                }
                for v in member_vertices
            ]

            # 2. Gather internal relationships between members
            community_edges = [
                e
                for e in graph.es
                if e["source_entity_id"] in member_set and e["target_entity_id"] in member_set
            ]
            relationships_data = [
                {
                    "source": e["source_entity_id"],
                    "relation_type": e["relation_type"],
                    "target": e["target_entity_id"],
                    "evidence_quotes": e.attributes().get("evidence_quotes", []),
                    "support_count": e.attributes().get("support_count", 1),
                }
                for e in community_edges
            ]

            # 3. Gather supporting text chunks
            supporting_chunk_ids: Set[str] = set()
            for v in member_vertices:
                supporting_chunk_ids.update(v.attributes().get("source_chunk_ids", []))
            for e in community_edges:
                supporting_chunk_ids.update(e.attributes().get("source_chunk_ids", []))

            supporting_chunks_text = ""
            if active_doc_store is not None and supporting_chunk_ids:
                chunks = active_doc_store.get_chunks_by_ids(sorted(list(supporting_chunk_ids)))
                supporting_chunks_text = "\n---\n".join(
                    f"[{c.chunk_id}]: {c.text}" for c in chunks
                )
            if not supporting_chunks_text:
                supporting_chunks_text = "No explicit supporting chunk text available."

            # 4. Generate summary: fast-path for singleton communities without internal edges
            if len(member_vertices) <= 1 and len(community_edges) == 0:
                if member_vertices:
                    v = member_vertices[0]
                    v_name = v.attributes().get("label", "") or v["entity_id"]
                    v_type = v.attributes().get("entity_type", "")
                    v_desc = v.attributes().get("description", "")
                    if v_desc:
                        summary_text = f"{v_name} ({v_type}): {v_desc}".strip()
                    else:
                        summary_text = f"Singleton community for entity {v_name} ({v_type})."
                else:
                    summary_text = f"Community {comm_id} containing members: {', '.join(member_ids)}."
            else:
                # Multi-entity or relational community: call LLM to synthesize relationships and themes
                prompt = (
                    self.prompt_template.replace("{community_id}", str(comm_id))
                    .replace("{entities_json}", json.dumps(entities_data, indent=2))
                    .replace("{relationships_json}", json.dumps(relationships_data, indent=2))
                    .replace("{supporting_chunks_text}", supporting_chunks_text)
                )

                try:
                    output: CommunitySummaryOutput = self.llm_client.generate_json(
                        prompt, CommunitySummaryOutput
                    )
                    summary_text = output.summary_text.strip()
                except Exception as e:
                    logger.warning("Failed LLM community summary for community %d: %s", comm_id, e)
                    # Grounded fallback summary from members
                    member_names = ", ".join(v.attributes().get("label", v["entity_id"]) for v in member_vertices)
                    summary_text = f"Community {comm_id} comprises {len(member_vertices)} entities: {member_names}."

            report_id = f"COMM_REPORT_{comm_id:04d}"
            target_kb_id = comm.get("kb_id") or (active_doc_store.kb_id if active_doc_store else "default_kb")

            report = CommunityReport(
                kb_id=target_kb_id,
                report_id=report_id,
                community_id=comm_id,
                partition_id=comm["partition_id"],
                member_entity_ids=member_ids,
                supporting_chunk_ids=sorted(list(supporting_chunk_ids)),
                summary_text=summary_text,
                model_tag=self.model_tag,
                prompt_version=self.prompt_version,
            )
            reports.append(report)

        # 5. Save to DocumentStore if available
        if active_doc_store is not None:
            active_doc_store.save_community_reports(reports)

        # 6. Index into ChromaVectorStore if available
        if self.vector_store is not None and self.embedding_provider is not None:
            try:
                texts = [r.summary_text for r in reports]
                embeddings = self.embedding_provider.embed_batch(texts)
                self.vector_store.upsert_community_reports(reports, embeddings)
            except Exception as e:
                logger.warning("Failed to embed or upsert community reports to vector store: %s", e)

        return reports
