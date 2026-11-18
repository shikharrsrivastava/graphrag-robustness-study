"""Generation subpackage providing Ollama-backed factual answer generation with citations."""

from src.generation.answer_generator import AnswerGenerationOutput, OllamaAnswerGenerator

__all__ = [
    "AnswerGenerationOutput",
    "OllamaAnswerGenerator",
]
