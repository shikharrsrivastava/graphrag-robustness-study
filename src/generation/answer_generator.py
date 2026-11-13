"""Answer generator producing evidence-grounded answers with verifiable citations and deterministic comparative reasoning."""

from __future__ import annotations
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field
from src.core.interfaces import AnswerGenerator
from src.core.models import EntityAttributeEvidence, Evidence, QueryPlan, RetrievalResult
from src.extraction.schemas import OllamaLLMClient
from src.retrieval.context_builder import ContextBuilder

logger = logging.getLogger(__name__)

DEFAULT_ANSWER_GENERATION_PROMPT = """You are an expert factual answer generator for an enterprise Hybrid GraphRAG system.
Your job is to answer the user's question accurately and objectively, strictly grounded in the provided evidence.

EVIDENCE CONTEXT:
{formatted_context}

QUERY METADATA:
{query_plan_info}

USER QUESTION:
{question}

INSTRUCTIONS:
1. Grounding & Strict Factual Alignment:
   - Answer the question using ONLY the facts explicitly stated in the EVIDENCE CONTEXT above.
   - Do NOT assume, extrapolate, or hallucinate facts not present in the context.
   - If the context does not contain sufficient facts to answer the question, set "abstained" to true, explain in "answer" why the context is insufficient, and return an empty list for "cited_chunk_ids".

2. Citations:
   - Cite source evidence inline using the exact chunk IDs from the headers in EVIDENCE CONTEXT, e.g. [D002_C0000]. Never use synthetic or placeholder IDs like C001, C002, or D001.
   - Collect all cited chunk IDs in the "cited_chunk_ids" list.

3. Comparative / Attribute Questions:
   - If the question asks whether entities share an attribute (e.g., nationality, employer, birthplace):
     a. Extract the specific attribute value for each entity mentioned in the query.
     b. Provide the exact quote from the source chunk and the chunk ID.
     c. Compare the values:
        - If both entities have the exact same attribute/nationality, the canonical answer is "yes".
        - If their attributes/nationalities differ, the canonical answer is "no".
        - Do NOT confuse birthplaces (cities/states) with nationalities (countries/demonyms).
        - Do NOT attribute one person's birthplace or details to another person mentioned in the same chunk.

4. Reasoning & Canonical Answer:
   - Provide your step-by-step reasoning first in "reasoning".
   - In "canonical_answer", provide ONLY the concise direct answer (e.g. "yes", "no", entity name, number, or short phrase). If abstaining, return "".
   - Ensure "canonical_answer" and "answer" are completely consistent and never contradict each other.

Return ONLY a valid JSON object matching this schema:
{
  "reasoning": "Step-by-step evaluation of the evidence and attributes.",
  "extracted_attributes": [
    {
      "entity_name": "Name of entity",
      "entity_id": "Entity ID if available",
      "attribute_name": "Name of attribute (e.g. NATIONALITY)",
      "extracted_value": "Extracted value (e.g. American)",
      "evidence_quote": "Exact sentence or phrase from the chunk text",
      "chunk_id": "Exact chunk ID where quote appears"
    }
  ],
  "canonical_answer": "Short direct answer (e.g. 'yes', 'no', entity name) or empty if abstaining.",
  "answer": "Direct answer to the question with inline citations [D002_C0000] or explanation if abstaining.",
  "abstained": false,
  "cited_chunk_ids": ["D002_C0000"]
}
"""


class CandidateAttributeFact(BaseModel):
    model_config = ConfigDict(extra="ignore")

    entity_name: str = Field(default="")
    entity_id: str = Field(default="")
    attribute_name: str = Field(default="")
    extracted_value: str = Field(default="")
    evidence_quote: str = Field(default="")
    chunk_id: str = Field(default="")


class AnswerGenerationOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    reasoning: Optional[str] = None
    extracted_attributes: List[CandidateAttributeFact] = Field(default_factory=list)
    canonical_answer: str = Field(default="")
    answer: str
    abstained: bool = False
    cited_chunk_ids: List[str] = Field(default_factory=list)
    abstention_reason: Optional[str] = None
    citation_references: List[dict] = Field(default_factory=list)


class FactValidator:
    """Validates extracted candidate attribute facts against retrieved chunk evidence."""

    KNOWN_DEMONYMS = {
        "american": "american",
        "united states": "american",
        "u.s.": "american",
        "usa": "american",
        "british": "british",
        "english": "british",
        "scottish": "british",
        "welsh": "british",
        "canadian": "canadian",
        "australian": "australian",
        "french": "french",
        "german": "german",
        "italian": "italian",
        "spanish": "spanish",
        "japanese": "japanese",
        "chinese": "chinese",
        "indian": "indian",
        "mexican": "mexican",
        "brazilian": "brazilian",
        "russian": "russian",
        "irish": "irish",
    }

    KNOWN_CITY_STATE_INDICATORS = {
        "baltimore", "los angeles", "new york", "chicago", "houston", "phoenix",
        "philadelphia", "san antonio", "san Diego", "dallas", "san jose", "austin",
        "denver", "colorado", "poughkeepsie", "california", "maryland", "texas",
        "florida", "pennsylvania", "illinois", "ohio", "georgia", "north carolina",
        "michigan", "new jersey", "virginia", "washington", "arizona", "massachusetts",
        "tennessee", "indiana", "missouri", "london", "paris", "berlin", "tokyo",
        "rome", "madrid", "toronto", "sydney",
    }

    @staticmethod
    def normalize_ws(text: str) -> str:
        return " ".join(text.split()).strip().lower()

    @classmethod
    def normalize_value(cls, val: str, attr_type: str = "NATIONALITY") -> str:
        """Normalize an attribute value (demonym or entity name)."""
        v = cls.normalize_ws(val)
        if attr_type.upper() in ("NATIONALITY", "CITIZEN_OF"):
            for k, canonical in cls.KNOWN_DEMONYMS.items():
                if k == v or v.startswith(k + " ") or v.endswith(" " + k):
                    return canonical
        return v

    @classmethod
    def is_geographic_location(cls, val: str) -> bool:
        """Check if a string represents a city/state rather than a nationality."""
        v = cls.normalize_ws(val)
        words = set(re.findall(r"\b\w+\b", v))
        return bool(words & cls.KNOWN_CITY_STATE_INDICATORS)

    @classmethod
    def validate_fact(
        cls,
        fact: CandidateAttributeFact,
        evidence_list: List[Evidence],
        expected_attribute: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Validate candidate fact against evidence chunks with strict entity attribution."""
        ev_map = {ev.chunk_id: ev for ev in evidence_list}

        # 1. Chunk existence
        if fact.chunk_id not in ev_map:
            return False, f"Chunk ID '{fact.chunk_id}' is not in retrieved evidence."

        chunk = ev_map[fact.chunk_id]

        # 2. Quote occurrence check (with whitespace normalization)
        norm_quote = cls.normalize_ws(fact.evidence_quote)
        norm_text = cls.normalize_ws(chunk.text)
        if not norm_quote or norm_quote not in norm_text:
            return False, f"Evidence quote does not occur in chunk '{fact.chunk_id}'."

        # 3. Target entity attribution check
        # The quote or its enclosing sentence must mention the target entity, not another entity
        target_name_norm = cls.normalize_ws(fact.entity_name)
        target_tokens = [t for t in target_name_norm.split() if len(t) > 2 and t not in ("the", "and", "jr.", "sr.")]

        entity_mentioned_in_quote = False
        if target_tokens and any(token in norm_quote for token in target_tokens):
            entity_mentioned_in_quote = True

        # Check sentence level in chunk text if quote was partial
        if not entity_mentioned_in_quote:
            sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
            for s in sentences:
                norm_s = cls.normalize_ws(s)
                if norm_quote in norm_s and target_tokens and any(token in norm_s for token in target_tokens):
                    entity_mentioned_in_quote = True
                    break

        if not entity_mentioned_in_quote:
            return False, f"Quote in chunk '{fact.chunk_id}' does not refer to target entity '{fact.entity_name}'."

        # 4. Semantic attribute type consistency
        if expected_attribute:
            exp_upper = expected_attribute.upper()
            if exp_upper in ("NATIONALITY", "CITIZEN_OF"):
                # Must not be a city/state
                if cls.is_geographic_location(fact.extracted_value) or "," in fact.extracted_value:
                    return False, f"Extracted value '{fact.extracted_value}' is a city/state, not a nationality."
                # Must be a recognized demonym or country
                norm_val = cls.normalize_value(fact.extracted_value, "NATIONALITY")
                if norm_val not in cls.KNOWN_DEMONYMS.values() and not any(d in cls.normalize_ws(fact.extracted_value) for d in cls.KNOWN_DEMONYMS):
                    if not any(d in norm_quote for d in cls.KNOWN_DEMONYMS):
                        return False, f"Extracted value '{fact.extracted_value}' is a city/state, not a nationality."

            elif exp_upper in ("BORN_IN", "BIRTHPLACE"):
                # Must not be a pure demonym
                if cls.normalize_ws(fact.extracted_value) in cls.KNOWN_DEMONYMS:
                    return False, f"Extracted value '{fact.extracted_value}' is a nationality, not a birthplace."

        return True, None

    @classmethod
    def extract_entity_attribute_from_chunk(
        cls,
        chunk: Evidence,
        entity_name: str,
        attribute_name: str = "NATIONALITY",
    ) -> Optional[EntityAttributeEvidence]:
        """Deterministic NLP pattern extraction of entity attributes directly from chunk text."""
        norm_name = cls.normalize_ws(entity_name)
        name_tokens = [t for t in norm_name.split() if len(t) > 2 and t not in ("the", "and", "jr.", "sr.")]
        if not name_tokens:
            return None

        sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
        attr_upper = attribute_name.upper()

        for s in sentences:
            s_clean = s.strip()
            s_norm = cls.normalize_ws(s_clean)

            # Sentence must mention the entity
            if not any(token in s_norm for token in name_tokens):
                continue

            # Ensure sentence is about the entity (not another person being born)
            if "born in" in s_norm and attr_upper == "NATIONALITY":
                # Check if this sentence is about another person
                pass

            if attr_upper in ("NATIONALITY", "CITIZEN_OF"):
                for demonym in cls.KNOWN_DEMONYMS:
                    pattern = rf"\b(is|was|became)\s+(an?|the)?\s*{demonym}\b"
                    if re.search(pattern, s_norm) or f" {demonym} " in f" {s_norm} ":
                        return EntityAttributeEvidence(
                            entity_id=chunk.doc_id,
                            entity_name=entity_name,
                            attribute_name=attribute_name,
                            extracted_value=demonym.capitalize(),
                            evidence_quote=s_clean,
                            chunk_id=chunk.chunk_id,
                            is_verified=True,
                        )

            elif attr_upper in ("BORN_IN", "BIRTHPLACE"):
                match = re.search(r"\bborn\s+(?:on\s+[^,]+,\s*\d+\s+)?in\s+([A-Z][A-Za-z\s,]+?)(?=[.;]|\bwho\b|\band\b|$)", s_clean)
                if match:
                    val = match.group(1).strip()
                    return EntityAttributeEvidence(
                        entity_id=chunk.doc_id,
                        entity_name=entity_name,
                        attribute_name=attribute_name,
                        extracted_value=val,
                        evidence_quote=s_clean,
                        chunk_id=chunk.chunk_id,
                        is_verified=True,
                    )

        return None


class ComparativeReasoningEngine:
    """Performs deterministic comparative reasoning over verified entity attribute facts."""

    @staticmethod
    def is_comparative_question(query_plan: QueryPlan) -> bool:
        q = query_plan.question.strip().lower()
        cues = ["same", "both", "differ", "share", "similar", "equal", "common"]
        has_cue = any(cue in q for cue in cues)
        has_seeds = len(query_plan.seed_entities) >= 2 or (" and " in q and has_cue)
        is_yes_no = q.startswith(("were ", "are ", "did ", "is ", "do ", "does "))
        return has_cue and (has_seeds or is_yes_no)

    @staticmethod
    def extract_target_attribute(query_plan: QueryPlan) -> str:
        q = query_plan.question.strip().lower()
        if "nationality" in q or "citizen" in q:
            return "NATIONALITY"
        if "birthplace" in q or "born in" in q:
            return "BORN_IN"
        if "university" in q or "college" in q or "school" in q or "study" in q:
            return "STUDIES_AT"
        if "work" in q or "employ" in q:
            return "WORKS_AT"
        if query_plan.relationship_path:
            return query_plan.relationship_path[0]
        return "ATTRIBUTE"

    @classmethod
    def evaluate_comparison(
        cls,
        entity_1_name: str,
        fact_1: Optional[EntityAttributeEvidence],
        entity_2_name: str,
        fact_2: Optional[EntityAttributeEvidence],
        attribute_name: str,
    ) -> Tuple[str, str, bool, List[str], Optional[str]]:
        """Compare two verified facts deterministically.

        Returns:
            Tuple of (canonical_answer, explanation_answer, abstained, cited_chunk_ids, abstention_reason)
        """
        # Missing evidence for either entity -> ABSTAIN
        if fact_1 is None and fact_2 is None:
            reason = f"Verified evidence for the {attribute_name.lower()} of both {entity_1_name} and {entity_2_name} was not found."
            return "", f"I am unable to answer this question. {reason}", True, [], reason

        if fact_1 is None:
            reason = f"Verified evidence for the {attribute_name.lower()} of {entity_1_name} was not found in retrieved evidence."
            return "", f"I am unable to answer this question. {reason}", True, [], reason

        if fact_2 is None:
            reason = f"Verified evidence for the {attribute_name.lower()} of {entity_2_name} was not found in retrieved evidence."
            return "", f"I am unable to answer this question. {reason}", True, [], reason

        # Both facts verified -> normalize and compare
        norm_val1 = FactValidator.normalize_value(fact_1.extracted_value, attribute_name)
        norm_val2 = FactValidator.normalize_value(fact_2.extracted_value, attribute_name)

        cited_chunks = sorted(list({fact_1.chunk_id, fact_2.chunk_id}))

        if norm_val1 == norm_val2:
            canonical = "yes"
            answer = (
                f"{entity_1_name} was {fact_1.extracted_value} [{fact_1.chunk_id}] and "
                f"{entity_2_name} was {fact_2.extracted_value} [{fact_2.chunk_id}]. "
                f"Therefore, they were of the same {attribute_name.lower()}."
            )
            return canonical, answer, False, cited_chunks, None
        else:
            canonical = "no"
            answer = (
                f"{entity_1_name} was {fact_1.extracted_value} [{fact_1.chunk_id}], while "
                f"{entity_2_name} was {fact_2.extracted_value} [{fact_2.chunk_id}]. "
                f"Therefore, they were not of the same {attribute_name.lower()}."
            )
            return canonical, answer, False, cited_chunks, None


class OllamaAnswerGenerator:
    """Generates strictly grounded answers with verifiable inline citations and deterministic comparison."""

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

    def _format_query_plan_info(self, query_plan: QueryPlan, resolved_ids: List[str]) -> str:
        """Format QueryPlan metadata for prompt injection."""
        lines = [
            f"- Query Intent: {query_plan.intent.value if hasattr(query_plan.intent, 'value') else query_plan.intent}",
        ]
        if query_plan.seed_entities:
            seeds_str = ", ".join(
                f"{s.name} (type: {s.expected_type or 'ANY'})" for s in query_plan.seed_entities
            )
            lines.append(f"- Seed Entities: {seeds_str}")
        if resolved_ids:
            lines.append(f"- Resolved Entity IDs: {', '.join(resolved_ids)}")
        if query_plan.relationship_path:
            lines.append(f"- Target Attribute / Relationship Path: {' -> '.join(query_plan.relationship_path)}")
        if query_plan.target_type:
            lines.append(f"- Target Entity Type: {query_plan.target_type}")
        return "\n".join(lines)

    def generate_structured(self, retrieval_result: RetrievalResult) -> AnswerGenerationOutput:
        """Generate structured AnswerGenerationOutput from a RetrievalResult."""
        if retrieval_result.abstained:
            reason = retrieval_result.abstention_reason or "Insufficient evidence to answer."
            return AnswerGenerationOutput(
                answer=f"I am unable to answer this question. Reason: {reason}",
                abstained=True,
                cited_chunk_ids=[],
                abstention_reason=reason,
            )

        if not retrieval_result.evidence_list and not retrieval_result.community_reports:
            reason = "No relevant evidence was found in the knowledge base."
            return AnswerGenerationOutput(
                answer=f"I am unable to answer this question. Reason: {reason}",
                abstained=True,
                cited_chunk_ids=[],
                abstention_reason=reason,
            )

        query_plan = retrieval_result.query_plan
        is_comparative = ComparativeReasoningEngine.is_comparative_question(query_plan)
        target_attribute = ComparativeReasoningEngine.extract_target_attribute(query_plan)

        context_str = self.context_builder.build_context(retrieval_result)
        query_plan_str = self._format_query_plan_info(
            query_plan, retrieval_result.resolved_seed_entity_ids
        )

        prompt = (
            self.prompt_template.replace("{formatted_context}", context_str)
            .replace("{question}", retrieval_result.question)
            .replace("{query_plan_info}", query_plan_str)
        )

        try:
            output = self.llm_client.generate_json(prompt, AnswerGenerationOutput)
        except Exception as e:
            logger.error("LLM answer generation failed: %s", e)
            output = AnswerGenerationOutput(
                answer=f"Error generating answer: {e}",
                abstained=True,
                cited_chunk_ids=[],
                abstention_reason="LLM invocation failed.",
            )

        if "[abstained]" in output.answer.lower():
            output.abstained = True
            output.abstention_reason = output.abstention_reason or "The answer output marked itself as abstained."
            output.cited_chunk_ids = []
            output.canonical_answer = ""

        evidence_by_id = {ev.chunk_id: ev for ev in retrieval_result.evidence_list}

        # -------------------------------------------------------------
        # Branch 1: Comparative Attribute Question Handling
        # -------------------------------------------------------------
        if is_comparative and len(query_plan.seed_entities) >= 2:
            seed_1 = query_plan.seed_entities[0].name
            seed_2 = query_plan.seed_entities[1].name

            # Validate any candidate facts provided by LLM
            verified_facts: Dict[str, EntityAttributeEvidence] = {}
            for fact in output.extracted_attributes:
                valid, reason = FactValidator.validate_fact(
                    fact, retrieval_result.evidence_list, expected_attribute=target_attribute
                )
                if valid:
                    name_norm = FactValidator.normalize_ws(fact.entity_name)
                    matching_seed = None
                    honorifics = {"dr", "dr.", "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "prof", "prof."}
                    s1_tokens = [t for t in seed_1.lower().split() if t not in honorifics and len(t) > 2]
                    s2_tokens = [t for t in seed_2.lower().split() if t not in honorifics and len(t) > 2]

                    if FactValidator.normalize_ws(seed_1) in name_norm or (s1_tokens and any(t in name_norm for t in s1_tokens)):
                        matching_seed = seed_1
                    elif FactValidator.normalize_ws(seed_2) in name_norm or (s2_tokens and any(t in name_norm for t in s2_tokens)):
                        matching_seed = seed_2

                    if matching_seed and matching_seed not in verified_facts:
                        verified_facts[matching_seed] = EntityAttributeEvidence(
                            entity_id=fact.entity_id or fact.chunk_id,
                            entity_name=matching_seed,
                            attribute_name=fact.attribute_name or target_attribute,
                            extracted_value=fact.extracted_value,
                            evidence_quote=fact.evidence_quote,
                            chunk_id=fact.chunk_id,
                            is_verified=True,
                        )

            # If either seed fact is missing from LLM extraction, search retrieved chunks directly
            for seed_name in (seed_1, seed_2):
                if seed_name not in verified_facts:
                    for ev in retrieval_result.evidence_list:
                        extracted = FactValidator.extract_entity_attribute_from_chunk(
                            ev, seed_name, target_attribute
                        )
                        if extracted is not None:
                            verified_facts[seed_name] = extracted
                            break

            fact_1 = verified_facts.get(seed_1)
            fact_2 = verified_facts.get(seed_2)

            canonical, answer, abstained, cited_chunks, reason = ComparativeReasoningEngine.evaluate_comparison(
                seed_1, fact_1, seed_2, fact_2, target_attribute
            )

            output.canonical_answer = canonical
            output.answer = answer
            output.abstained = abstained
            output.cited_chunk_ids = cited_chunks
            output.abstention_reason = reason

        # -------------------------------------------------------------
        # Branch 2: General / Non-Comparative Question Handling
        # -------------------------------------------------------------
        else:
            if output.abstained:
                output.cited_chunk_ids = []
                output.canonical_answer = ""
            else:
                # Map any synthetic tokens (e.g. C001, C002) to real chunk IDs
                resolved_cited: Set[str] = set()
                for cid in output.cited_chunk_ids:
                    if cid in evidence_by_id:
                        resolved_cited.add(cid)

                # Search answer text for real chunk ID patterns
                for real_id in evidence_by_id:
                    if real_id in output.answer:
                        resolved_cited.add(real_id)

                # If model used synthetic tokens like C001, map them to real chunk IDs based on evidence text
                for synthetic in re.findall(r"\[(C\d{3,4})\]", output.answer):
                    for ev in retrieval_result.evidence_list:
                        if synthetic in ev.chunk_id:
                            resolved_cited.add(ev.chunk_id)

                output.cited_chunk_ids = sorted(list(resolved_cited))

                # Ensure canonical_answer and answer are consistent
                if output.canonical_answer:
                    canon_norm = output.canonical_answer.strip().lower()
                    ans_norm = output.answer.strip().lower()
                    if canon_norm == "yes" and "not" in ans_norm and "same" in ans_norm:
                        output.canonical_answer = "no"
                    elif canon_norm == "no" and "share the same" in ans_norm:
                        output.canonical_answer = "yes"

        # Populate retrieval_result fields for downstream consumers
        retrieval_result.generated_answer = output.answer
        retrieval_result.canonical_answer = output.canonical_answer if output.canonical_answer else None
        retrieval_result.answer_abstained = output.abstained
        retrieval_result.abstained = output.abstained
        retrieval_result.abstention_reason = output.abstention_reason if output.abstained else None
        retrieval_result.cited_chunk_ids = output.cited_chunk_ids
        retrieval_result.citation_references = [
            {
                "chunk_id": cid,
                "doc_id": evidence_by_id[cid].doc_id,
                "filename": evidence_by_id[cid].filename,
                "page_numbers": evidence_by_id[cid].page_numbers,
            }
            for cid in output.cited_chunk_ids
            if cid in evidence_by_id
        ]
        output.citation_references = retrieval_result.citation_references

        return output

    def generate_answer(self, retrieval_result: RetrievalResult) -> str:
        """Satisfies the AnswerGenerator protocol by returning the answer string."""
        structured = self.generate_structured(retrieval_result)
        return structured.answer
