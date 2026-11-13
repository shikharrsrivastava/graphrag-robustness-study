"""Pydantic schemas for raw LLM extraction responses and lightweight Ollama client."""

from __future__ import annotations
import json
import logging
import time
import requests
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from src.core.configuration import OllamaConfig
from src.core.models import Polarity

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class RawEntityMentionItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    surface_name: str = Field(..., min_length=1)
    entity_type: str = Field(..., min_length=1)
    description: str = Field(default="")


class RawEntityExtractionOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    entities: List[RawEntityMentionItem] = Field(default_factory=list)


class RawRelationshipItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source_mention_id: str = Field(..., min_length=1)
    target_mention_id: str = Field(..., min_length=1)
    relation_type: str = Field(..., min_length=1)
    evidence_quote: str = Field(..., min_length=1)
    polarity: Polarity = Field(default=Polarity.POSITIVE)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("polarity", mode="before")
    @classmethod
    def normalize_polarity(cls, v: Any) -> Polarity:
        if isinstance(v, Polarity):
            return v
        if isinstance(v, str):
            clean = v.strip().upper()
            if clean in Polarity.__members__:
                return Polarity[clean]
        return Polarity.POSITIVE


class RawRelationshipExtractionOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    relationships: List[RawRelationshipItem] = Field(default_factory=list)


def clean_llm_json_text(text: str) -> str:
    """Clean markdown backticks, fences, and extraneous pre/post text from raw LLM text."""
    cleaned = text.strip()
    if "```json" in cleaned:
        start = cleaned.find("```json") + 7
        end = cleaned.rfind("```")
        if end > start:
            cleaned = cleaned[start:end].strip()
    elif "```" in cleaned:
        start = cleaned.find("```") + 3
        end = cleaned.rfind("```")
        if end > start:
            cleaned = cleaned[start:end].strip()
    else:
        # Extract the outermost JSON object { ... } or array [ ... ]
        first_curly = cleaned.find("{")
        first_bracket = cleaned.find("[")
        candidates_first = [pos for pos in [first_curly, first_bracket] if pos != -1]
        if candidates_first:
            first_pos = min(candidates_first)
            last_curly = cleaned.rfind("}")
            last_bracket = cleaned.rfind("]")
            candidates_last = [pos for pos in [last_curly, last_bracket] if pos != -1]
            if candidates_last:
                last_pos = max(candidates_last)
                if last_pos > first_pos:
                    cleaned = cleaned[first_pos : last_pos + 1].strip()

    return cleaned


class OllamaLLMClient:
    """Client for Ollama REST API /api/generate with bounded retries and mock support."""

    def __init__(
        self,
        config: Optional[OllamaConfig] = None,
        custom_generator: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.config = config or OllamaConfig()
        self.custom_generator = custom_generator
        self.stats: Dict[str, Any] = {"requests": 0, "failed_requests": 0, "prompt_chars": [],
                                     "failed_generations": 0, "request_seconds": [],
                                     "response_total_duration_ns": []}

    def generate(self, prompt: str, format_json: bool = False) -> str:
        """Call Ollama generate endpoint or custom generator."""
        started = time.perf_counter()
        self.stats["requests"] += 1
        self.stats["prompt_chars"].append(len(prompt))
        if self.custom_generator is not None:
            try:
                return self.custom_generator(prompt)
            except Exception:
                self.stats["failed_requests"] += 1
                raise
            finally:
                self.stats["request_seconds"].append(time.perf_counter() - started)

        url = f"{self.config.base_url.rstrip('/')}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.config.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
            },
        }
        if format_json:
            payload["format"] = "json"

        try:
            response = requests.post(url, json=payload, timeout=self.config.timeout_seconds)
            response.raise_for_status()
            data = response.json()
            if isinstance(data.get("total_duration"), (int, float)):
                self.stats["response_total_duration_ns"].append(data["total_duration"])
            return data.get("response", "")
        except Exception:
            self.stats["failed_requests"] += 1
            raise
        finally:
            self.stats["request_seconds"].append(time.perf_counter() - started)

    def generate_json(
        self,
        prompt: str,
        schema_cls: Type[T],
        max_retries: Optional[int] = None,
    ) -> T:
        """Generate and validate JSON output against schema_cls with bounded retries."""
        retries = max_retries if max_retries is not None else self.config.max_retries
        attempts = 0
        last_error: Optional[Exception] = None

        while attempts <= retries:
            attempts += 1
            # Alternate format: json and unconstrained (gemma3 excels with unconstrained markdown fences)
            use_format_json = attempts % 2 == 1 and "gemma3" not in self.config.llm_model.lower()
            try:
                raw_text = self.generate(prompt, format_json=use_format_json)
                cleaned = clean_llm_json_text(raw_text)
                if not cleaned or cleaned == "{}":
                    # Fallback to unconstrained call if format_json returned empty object
                    raw_text = self.generate(prompt, format_json=False)
                    cleaned = clean_llm_json_text(raw_text)

                parsed = json.loads(cleaned)
                return schema_cls.model_validate(parsed)
            except Exception as e:
                last_error = e
                self.stats["failed_generations"] += 1
                if isinstance(e, requests.Timeout):
                    category = "timeout"
                elif isinstance(e, (requests.ConnectionError,)):
                    category = "connection"
                elif isinstance(e, requests.HTTPError):
                    status = getattr(getattr(e, "response", None), "status_code", None)
                    category = "http_transient" if status == 429 or (status is not None and status >= 500) else "http_permanent"
                elif isinstance(e, (json.JSONDecodeError, ValidationError, ValueError)):
                    category = "malformed_or_invalid_json"
                else:
                    category = "non_transient"
                logger.warning(
                    "LLM JSON generation attempt %d/%d failed for %s (%s): %s",
                    attempts,
                    retries + 1,
                    schema_cls.__name__,
                    category,
                    e,
                )
                transient = isinstance(e, (requests.Timeout, requests.ConnectionError)) or category in {"malformed_or_invalid_json", "http_transient"}
                if attempts <= retries and transient:
                    time.sleep(min(0.25 * attempts, 0.75))
                elif not transient:
                    break
                if attempts > retries:
                    break

        raise ValueError(
            f"Failed to generate valid JSON matching {schema_cls.__name__} after {attempts} attempts. Last error: {last_error}"
        )
