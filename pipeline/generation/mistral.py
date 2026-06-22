"""
RAGScope — Mistral 7B Client
============================
Concrete LLM client for Mistral AI's Mistral 7B Instruct v0.1 served via Ollama.

Used in Conditions C (Dense) and D (Hybrid) of the 2×2 factorial experiment.

Model card: https://ollama.com/library/mistral
"""

from __future__ import annotations

from pipeline.generation.llm_client import BaseLLMClient


class MistralClient(BaseLLMClient):
    """
    LLM client for Mistral 7B Instruct.

    Mistral Instruct uses the ``[INST]`` / ``[/INST]`` chat template.
    We override ``_build_prompt`` to produce a properly formatted prompt.
    """

    @property
    def model_name(self) -> str:
        return "mistral"

    def _build_prompt(self, query: str, context_chunks: list[str]) -> str:
        """Mistral Instruct [INST] chat template."""
        context = "\n\n---\n\n".join(
            f"Passage {i + 1}:\n{chunk}" for i, chunk in enumerate(context_chunks)
        )
        return f"<s>[INST] {self.SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query} [/INST]"
