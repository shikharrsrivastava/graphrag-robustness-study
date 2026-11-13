"""Answer generator invoking Ollama LLM to produce strictly grounded answers with inline citations."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field
from src.core.interfaces import AnswerGenerator
from src.core.models import RetrievalResult
from src.extraction.schemas import OllamaLLMClient
from src.retrieval.context_builder import ContextBuilder

logger = logging.getLogger(__name__)

DEFAULT_ANSWER_GENERATION_PROMPT = """You are an expert factual answer generator for an enterprise Hybrid GraphRAG system.
Your job is to answer the user's question accurately and objectively, strictly grounded in the provided evidence.

EVIDENCE CONTEXT:
{formatted_context}

USER QUESTION:
{question}

INSTRUCTIONS:
1. Grounding & Strict Factual Alignment:
   - Answer the question using ONLY the facts explicitly stated in the EVIDENCE CONTEXT above.
   - Do NOT assume, extrapolate, or hallucinate facts not present in the context.
   - If the context does not contain sufficient facts to answer the question, you MUST abstain: set "abstained" to true, explain in "answer" why the context is insufficient, and return an empty list for "cited_chunk_ids".

2. Citations:
   - Every factual claim in your answer MUST cite its source inline using the format [chunk_id, doc_id, p.X] (or [chunk_id, doc_id] if page numbers are not available). Example: "Alice is supervised by Dr. Smith [C001, D001, p.1]."
   - Collect all cited chunk IDs in the "cited_chunk_ids" list.

3. Handling Conflicting Evidence:
   - If different parts of the context contradict each other, explicitly mention the conflicting viewpoints and cite both sources.

4. Short Canonical Answer:
   - In "canonical_answer", provide ONLY the concise direct answer (e.g., "yes", "no", entity name, number, or short phrase) without citations or filler. For yes/no questions, output strictly "yes" or "no". If abstaining, return "".

Return ONLY a valid JSON object matching this schema:
{
  "canonical_answer": "Short direct answer (e.g. 'yes', 'no', entity name) or empty if abstaining.",
  "answer": "Direct answer to the question with inline citations [C001, D001, p.1] or explanation if abstaining.",
  "abstained": false,
  "cited_chunk_ids": ["C001"]
}
"""


class AnswerGenerationOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    canonical_answer: str = Field(default="")
    answer: str
    abstained: bool = False
    cited_chunk_ids: List[str] = Field(default_factory=list)
    abstention_reason: Optional[str] = None
    citation_references: List[dict] = Field(default_factory=list)


class OllamaAnswerGenerator:
    """Generates strictly grounded answers with inline citations using OllamaLLMClient."""

    def __init__(
        self,
        llm_client: OllamaLLMClient,
        prompt_path: Optional[Union[str, Path]] = None,
        context_builder: Optional[ContextBuilder] = None,
    ) -> None:
        self.llm_client = llm_client
        self.context_builder = context_builder or ContextBuilder()

        if prompt_path is not None and Path(prompt_path).exists():
            self.prompt_template = Path(prompt_path).read_text(encoding="utf-8")
        else:
            default_path = Path("prompts/answer_generation.txt")
            if default_path.exists():
                self.prompt_template = default_path.read_text(encoding="utf-8")
            else:
                self.prompt_template = DEFAULT_ANSWER_GENERATION_PROMPT

    def generate_structured(self, retrieval_result: RetrievalResult) -> AnswerGenerationOutput:
        """Generate structured AnswerGenerationOutput from a RetrievalResult."""
        if retrieval_result.abstained:
            reason = retrieval_result.abstention_reason or "Insufficient evidence to answer."
            return AnswerGenerationOutput(
                answer=f"I am unable to answer this question. Reason: {reason}",
                abstained=True,
                cited_chunk_ids=[],
            )

        if not retrieval_result.evidence_list and not retrieval_result.community_reports:
            return AnswerGenerationOutput(
                answer="I am unable to answer this question. Reason: No relevant evidence was found in the knowledge base.",
                abstained=True,
                cited_chunk_ids=[],
            )

        context_str = self.context_builder.build_context(retrieval_result)
        prompt = self.prompt_template.replace(
            "{formatted_context}", context_str
        ).replace("{question}", retrieval_result.question)

        try:
            output = self.llm_client.generate_json(prompt, AnswerGenerationOutput)
            if "[abstained]" in output.answer.lower():
                output.abstained = True
                output.abstention_reason = output.abstention_reason or "The answer output marked itself as abstained."
                output.cited_chunk_ids = []
            allowed = {ev.chunk_id for ev in retrieval_result.evidence_list}
            output.cited_chunk_ids = [cid for cid in output.cited_chunk_ids if cid in allowed]
            retrieval_result.generated_answer = output.answer
            retrieval_result.canonical_answer = output.canonical_answer.strip() if output.canonical_answer else None
            retrieval_result.answer_abstained = output.abstained
            retrieval_result.abstained = output.abstained
            retrieval_result.abstention_reason = output.abstention_reason if output.abstained else None
            evidence_by_id = {ev.chunk_id: ev for ev in retrieval_result.evidence_list}
            retrieval_result.cited_chunk_ids = output.cited_chunk_ids
            retrieval_result.citation_references = [
                {"chunk_id": cid, "doc_id": evidence_by_id[cid].doc_id,
                 "filename": evidence_by_id[cid].filename,
                 "page_numbers": evidence_by_id[cid].page_numbers}
                for cid in output.cited_chunk_ids
            ]
            return output
        except Exception as e:
            logger.error("Failed to generate structured answer: %s", e)
            return AnswerGenerationOutput(
                answer=f"Error generating answer: {e}",
                abstained=True,
                cited_chunk_ids=[],
                abstention_reason="Answer generation failed.",
            )

    def generate_answer(self, retrieval_result: RetrievalResult) -> str:
        """Satisfies the AnswerGenerator protocol by returning the answer string."""
        structured = self.generate_structured(retrieval_result)
        return structured.answer
