"""Context builder assembling and formatting hybrid GraphRAG evidence under a character budget."""

from __future__ import annotations
import logging
from typing import List, Union
from src.core.models import RetrievalResult

logger = logging.getLogger(__name__)

TRUSTED_COMMUNITY_STATUSES = frozenset({"VERIFIED", "FALLBACK"})


class ContextBuilder:
    """Formats community summaries, graph paths, and chunk evidence within a strict character budget."""

    def __init__(
        self,
        max_chars: int = 12000,
        allow_unverified_communities: bool = False,
        graph_context_mode: Union[bool, str] = False,
    ) -> None:
        if max_chars < 500:
            raise ValueError("max_chars must be at least 500 characters.")
        self.max_chars = max_chars
        self.allow_unverified_communities = allow_unverified_communities
        self.graph_context_mode = graph_context_mode

    def build_context(self, retrieval_result: RetrievalResult) -> str:
        """Construct structured markdown context from a RetrievalResult within max_chars."""
        sections: List[str] = []
        current_len = 0

        mode_str = str(self.graph_context_mode).strip().lower()
        is_none_mode = mode_str == "none"
        # The string "off" omits the graph section (path lines carry provenance marker text that the
        # generator copies into answers). Boolean False keeps the legacy path block for backward
        # compatibility with existing configurations and tests.
        is_off_mode = isinstance(self.graph_context_mode, str) and mode_str == "off"
        is_fix_e_mode = (self.graph_context_mode is True) or (mode_str in ("true", "fix_e", "edges"))

        # 1. Community Summaries (filtered using allowlist: only VERIFIED and FALLBACK trusted by default)
        if not is_none_mode and retrieval_result.community_reports:
            comm_lines: List[str] = []
            for rep in retrieval_result.community_reports:
                status = getattr(rep, "verification_status", None)
                if status not in TRUSTED_COMMUNITY_STATUSES and not self.allow_unverified_communities:
                    logger.warning(
                        "Omitting community report %s (community %d) with untrusted verification status '%s' from generation context.",
                        rep.report_id,
                        rep.community_id,
                        status,
                    )
                    continue
                comm_lines.append(f"- [Community {rep.community_id}]: {rep.summary_text.strip()}")

            if comm_lines:
                comm_block = "### Community Summaries\n" + "\n".join(comm_lines)
                if current_len + len(comm_block) + 2 <= self.max_chars:
                    sections.append(comm_block)
                    current_len += len(comm_block) + 2

        # 2. Supporting Graph Traversal Paths / Extracted Graph Relationships
        if is_none_mode or is_off_mode:
            # Modes "none" and "off": the graph is used to retrieve evidence chunks, but NO graph section
            # is sent to the generator ("off" still includes trusted community summaries; "none" does not).
            pass
        elif is_fix_e_mode:
            # Fix E(a): Include ONLY actual relationship edges, each with its chunk ID,
            # labeled as extracted facts that may be incomplete. Clarify missing facts do not mean false.
            # Omit GRAPH_SEED_PROVENANCE and other bookkeeping lines.
            edge_lines: List[str] = []
            seen_edges = set()
            for ev in retrieval_result.evidence_list:
                for p in ev.supporting_graph_paths:
                    if "-[" in p and "]->" in p:
                        # Actual edge
                        entry = f"- [Chunk: {ev.chunk_id}] {p}"
                        if entry not in seen_edges:
                            seen_edges.add(entry)
                            edge_lines.append(entry)

            if edge_lines:
                path_block = (
                    "### Extracted Graph Relationships (Incomplete Extracted Facts)\n"
                    "Note: The knowledge graph contains extracted relationship edges which may be incomplete. "
                    "A fact not present in the graph does NOT mean it is false; primary evidence chunks remain authoritative.\n"
                    + "\n".join(edge_lines)
                )
                if current_len + len(path_block) + 2 <= self.max_chars:
                    sections.append(path_block)
                    current_len += len(path_block) + 2
        else:
            all_paths: List[str] = []
            for ev in retrieval_result.evidence_list:
                for p in ev.supporting_graph_paths:
                    if p not in all_paths:
                        all_paths.append(p)

            if all_paths:
                path_lines = ["### Supporting Graph Traversal Paths"]
                for path in all_paths:
                    path_lines.append(f"- {path}")
                path_block = "\n".join(path_lines)
                if current_len + len(path_block) + 2 <= self.max_chars:
                    sections.append(path_block)
                    current_len += len(path_block) + 2

        # 3. Primary Evidence Chunks (ordered by RRF score)
        chunk_blocks: List[str] = []
        for ev in retrieval_result.evidence_list:
            pages_str = ", ".join(str(p) for p in ev.page_numbers) if ev.page_numbers else "N/A"
            routes_str = "+".join(r.value for r in ev.routes)
            doc_info = f"{ev.doc_id} ({ev.filename})" if ev.filename else ev.doc_id

            header = (
                f"### [Chunk: {ev.chunk_id} | Doc: {doc_info} | Pages: {pages_str} | Routes: {routes_str}]"
            )
            block = f"{header}\n{ev.text.strip()}"

            # Check character budget
            if current_len + len(block) + 2 <= self.max_chars:
                chunk_blocks.append(block)
                current_len += len(block) + 2
            else:
                # If budget exceeded and we haven't added any chunk yet, add truncated chunk
                if not chunk_blocks:
                    remaining = self.max_chars - current_len - len(header) - 10
                    if remaining > 50:
                        truncated_text = ev.text.strip()[:remaining] + "..."
                        chunk_blocks.append(f"{header}\n{truncated_text}")
                break

        if chunk_blocks:
            sections.append("### Primary Evidence Chunks\n\n" + "\n\n".join(chunk_blocks))

        return "\n\n".join(sections)
