"""
RAGScope — Token Counter
=========================
Counts prompt and completion tokens for each query and computes
estimated cost based on configurable per-token rates.

Token counting uses tiktoken (cl100k_base) — the same tokeniser used
during chunking — ensuring consistent counts across the pipeline.
For local Ollama inference the cost will always be $0.00, but the
counter is designed to support cloud model cost estimation if the
research is extended to compare local vs. API-hosted models.
"""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken

from config.settings import settings


@dataclass
class TokenUsage:
    """
    Token usage record for a single query execution.

    Attributes
    ----------
    prompt_tokens      : Tokens in the full prompt (query + context passages).
    completion_tokens  : Tokens in the generated response.
    total_tokens       : prompt_tokens + completion_tokens.
    estimated_cost_usd : Cost estimate in USD.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
        }


class TokenCounter:
    """
    Counts tokens and estimates cost for RAG pipeline calls.

    Parameters
    ----------
    encoding_name : tiktoken encoding. Default ``cl100k_base``.
    """

    _ENCODING_NAME = "cl100k_base"

    def __init__(self, encoding_name: str = _ENCODING_NAME) -> None:
        self._enc = tiktoken.get_encoding(encoding_name)

    def build_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> TokenUsage:
        """
        Build a ``TokenUsage`` record with cost estimation.

        Parameters
        ----------
        prompt_tokens     : Token count of the input prompt.
        completion_tokens : Token count of the generated response.
        """
        cost = (prompt_tokens / 1000) * settings.cost_per_1k_prompt_tokens + (
            completion_tokens / 1000
        ) * settings.cost_per_1k_completion_tokens
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            estimated_cost_usd=round(cost, 6),
        )

    def from_generation_response(self, response) -> TokenUsage:
        """
        Build a ``TokenUsage`` directly from a ``GenerationResponse`` object.
        Ollama provides token counts in the response — this avoids re-counting.
        """
        return self.build_usage(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
