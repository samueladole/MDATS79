"""
RAGScope — LLM Client Base
============================
Base class and shared logic for all LLM generation backends.
Concrete implementations (Llama3Client, MistralClient) inherit from this.

All clients communicate with the Ollama HTTP API, which serves models
locally. No external API keys or network calls leave the research environment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import httpx
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings

# ── Response type ─────────────────────────────────────────────────────────────


@dataclass
class GenerationResponse:
    """
    Structured response from an LLM generation call.

    Attributes
    ----------
    text              : Generated response text.
    model             : Model name that produced the response.
    prompt_tokens     : Tokens in the input prompt.
    completion_tokens : Tokens in the generated response.
    total_tokens      : Sum of prompt + completion tokens.
    generation_ms     : Wall-clock time for the generation call (ms).
    raw               : Raw response dict from the Ollama API.
    """

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    generation_ms: float
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def estimated_cost(self) -> float:
        """
        Estimated cost in USD based on the configured per-token rates.
        Returns 0.00 for fully local inference (the research default).
        """
        prompt_cost = (self.prompt_tokens / 1000) * settings.cost_per_1k_prompt_tokens
        completion_cost = (self.completion_tokens / 1000) * settings.cost_per_1k_completion_tokens
        return round(prompt_cost + completion_cost, 6)


# ── Base client ───────────────────────────────────────────────────────────────


class BaseLLMClient(ABC):
    """
    Abstract base class for Ollama-backed LLM clients.

    Subclasses must implement ``model_name`` and may override
    ``_build_prompt`` for model-specific prompt formatting.
    """

    # ── System prompt for RAG generation ─────────────────────────────────────
    SYSTEM_PROMPT = (
        "You are a precise and factual question-answering assistant. "
        "Answer the question using ONLY the information provided in the context below. "
        "If the context does not contain enough information to answer confidently, "
        "state that clearly. Do not use any prior knowledge or make assumptions "
        "beyond what is explicitly stated in the context. "
        "Keep your answer concise and directly address the question." \
        "\n\nIMPORTANT: Only use the provided context. Do not use outside knowledge."
    )

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._timeout = timeout or settings.ollama_timeout_seconds
        self._client = httpx.Client(timeout=self._timeout)

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Ollama model identifier, e.g. 'llama3:8b'."""
        ...

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        context_chunks: list[str],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> GenerationResponse:
        """
        Generate a response to ``query`` grounded in ``context_chunks``.

        Parameters
        ----------
        query          : The user question.
        context_chunks : List of retrieved passage texts to use as context.
        max_tokens     : Maximum tokens in the generated response.
        temperature    : Sampling temperature. 0.0 = greedy (deterministic).

        Returns
        -------
        GenerationResponse
        """
        prompt = self._build_prompt(query, context_chunks)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "seed": settings.experiment_random_seed,
            },
        }

        import time

        t0 = time.perf_counter()
        raw = self._call_ollama(payload)
        generation_ms = (time.perf_counter() - t0) * 1000

        text = raw.get("response", "").strip()

        # Ollama returns eval_count (completion tokens) and prompt_eval_count.
        completion_tokens = raw.get("eval_count") or raw.get("completion_tokens") or 0
        prompt_tokens = raw.get("prompt_eval_count") or raw.get("prompt_tokens") or 0

        # Warn if the expected eval_count is missing, but still return a response.
        if "eval_count" not in raw:
            logger.warning("Missing eval_count in Ollama response")

        return GenerationResponse(
            text=text,
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            generation_ms=round(generation_ms, 2),
            raw=raw,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_prompt(self, query: str, context_chunks: list[str]) -> str:
        """
        Build a RAG prompt with stronger grounding, better structure,
        and controlled context size by injecting context before the question.
        Override in subclasses if the model requires a specific template.
        """

        context = "\n\n".join(
            f"<<<PASSAGE {i + 1}>>>\n{chunk}\n<<<END PASSAGE>>>"
            for i, chunk in enumerate(context_chunks)
        )

        return (
            f"[SYSTEM]\n"
            f"{self.SYSTEM_PROMPT}\n\n"
            f"[RETRIEVED CONTEXT - USE ONLY THIS]\n"
            f"{context}\n\n"
            f"[QUESTION]\n"
            f"{query}\n\n"
            f"Instruction: Answer ONLY using the evidence above. "
            f"If the answer is not present, say 'Not found in provided context.'\n\n"
            f"[ANSWER]\n"
        )

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _call_ollama(self, payload: dict) -> dict:
        """POST to the Ollama /api/generate endpoint with retry logic."""
        logger.debug(
            f"Calling Ollama model '{self.model_name}' "
            f"(max_tokens={payload['options']['num_predict']}) …"
        )
        response = self._client.post(
            f"{self._base_url}/api/generate",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model='{self.model_name}')"
