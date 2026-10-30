"""
Generation clients.

``build_llm_client`` maps an Ollama model tag to its client. The two research
models keep their own classes, because their hand-written chat templates are
part of the published experimental setup. Any other tag (for example a small
CPU-friendly model such as ``llama3.2:3b``) uses the generic prompt in
``BaseLLMClient``.
"""

from __future__ import annotations

from pipeline.generation.llm_client import BaseLLMClient, GenerationResponse

__all__ = ["BaseLLMClient", "GenerationResponse", "OllamaClient", "build_llm_client"]


class OllamaClient(BaseLLMClient):
    """Client for any Ollama model tag, using the generic RAG prompt."""

    def __init__(self, model_tag: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._model_tag = model_tag

    @property
    def model_name(self) -> str:
        return self._model_tag


def build_llm_client(model_tag: str) -> BaseLLMClient:
    """Return the client for ``model_tag``."""
    if model_tag == "llama3":
        from pipeline.generation.llama3 import Llama3Client

        return Llama3Client()
    if model_tag == "mistral":
        from pipeline.generation.mistral import MistralClient

        return MistralClient()
    return OllamaClient(model_tag)
