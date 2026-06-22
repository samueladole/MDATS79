"""
RAGScope — Llama 3 (8B) Client
================================
Concrete LLM client for Meta's Llama 3 8B Instruct model served via Ollama.

Used in Conditions A (Dense) and B (Hybrid) of the 2×2 factorial experiment.
Also used as the RAGAS judge model for metric computation.

Model card: https://ollama.com/library/llama3
"""

from __future__ import annotations

from pipeline.generation.llm_client import BaseLLMClient


class Llama3Client(BaseLLMClient):
    """
    LLM client for Llama 3 1B Instruct.

    Llama 3 uses the ``<|begin_of_text|>`` / ``<|eot_id|>`` chat template.
    We override ``_build_prompt`` to produce a properly formatted instruct
    prompt that Ollama's Llama 3 backend expects, which improves faithfulness
    compared to the generic completion-style prompt.
    """

    @property
    def model_name(self) -> str:
        return "llama3"

    def _build_prompt(self, query: str, context_chunks: list[str]) -> str:
        """Llama 3 Instruct chat template."""
        context = "\n\n---\n\n".join(
            f"Passage {i + 1}:\n{chunk}" for i, chunk in enumerate(context_chunks)
        )
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{self.SYSTEM_PROMPT}"
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"Context:\n{context}\n\nQuestion: {query}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
