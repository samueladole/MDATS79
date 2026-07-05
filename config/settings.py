"""
RAGScope — Centralised Configuration
=====================================
All configuration is read from environment variables (or a .env file).
Every module imports from here; nothing reads os.environ directly.

Usage
-----
    from config.settings import settings
    print(settings.ollama_base_url)
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Project-wide settings loaded from the .env file.
    Pydantic validates types and raises clear errors on misconfiguration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Ollama ─────────────────────────────────────────────────────────────────
    ollama_base_url: str = Field(
        default="http://host.docker.internal:12434",
        description="Base URL for the Ollama inference server.",
    )
    ollama_port: int = Field(default=12434)
    ollama_timeout_seconds: int = Field(
        default=120,
        description="Seconds to wait for a single LLM generation response.",
    )
    default_llm: str = Field(
        default="llama3",
        description="Default LLM model name for interactive queries.",
    )

    # ── Retrieval ──────────────────────────────────────────────────────────────
    retrieval_strategy: str = Field(
        default="hybrid",
        description="Retrieval strategy: 'dense' or 'hybrid'.",
    )
    top_k: int = Field(
        default=5,
        description="Number of passages returned per query.",
    )
    hybrid_dense_weight: float = Field(
        default=0.6,
        description="Weight given to dense scores in hybrid retrieval (0–1).",
    )

    # ── Embeddings ─────────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="SentenceTransformer model name for embedding generation.",
    )
    embedding_device: str = Field(
        default="cpu",
        description="Device for embedding inference: 'cpu', 'cuda', or 'mps'.",
    )

    # ── ChromaDB ───────────────────────────────────────────────────────────────
    chroma_host: str = Field(default="localhost")
    chroma_port: int = Field(default=8000)
    chroma_collection: str = Field(default="ragscope_corpus")

    # ── RAGAS ──────────────────────────────────────────────────────────────────
    ragas_judge_model: str = Field(
        default="llama3",
        description="LLM used as the RAGAS judge for metric computation.",
    )
    ragas_max_tokens: int = Field(default=2048)

    # ── Cost estimation ────────────────────────────────────────────────────────
    cost_per_1k_prompt_tokens: float = Field(default=0.00)
    cost_per_1k_completion_tokens: float = Field(default=0.00)
    cost_currency: str = Field(default="USD")

    # ── Telemetry ──────────────────────────────────────────────────────────────
    telemetry_store_dir: Path = Field(default=Path("telemetry/store"))
    telemetry_max_records: int = Field(default=1000)

    # ── Dashboard (Streamlit) ──────────────────────────────────────────────────
    streamlit_server_address: str = Field(default="0.0.0.0")
    streamlit_server_port: int = Field(default=8501)

    # ── Experiment ─────────────────────────────────────────────────────────────
    experiment_random_seed: int = Field(default=42)
    dev_cache_llm_responses: bool = Field(default=False)
    dev_cache_dir: Path = Field(default=Path(".cache/llm_responses"))

    # ── Chunking ───────────────────────────────────────────────────────────────
    chunk_size: int = Field(
        default=512,
        description="Maximum number of tokens per document chunk.",
    )
    chunk_overlap: int = Field(
        default=64,
        description="Token overlap between consecutive chunks.",
    )

    # ── Logging ────────────────────────────────────────────────────────────────
    log_level: str = Field(default="DEBUG", description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    log_format: str = Field(
        default="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} — {message}"
    )

    # ── Derived properties ─────────────────────────────────────────────────────
    @property
    def chroma_url(self) -> str:
        return f"http://{self.chroma_host}:{self.chroma_port}"

    @field_validator("retrieval_strategy")
    @classmethod
    def validate_retrieval_strategy(cls, v: str) -> str:
        allowed = {"dense", "hybrid"}
        if v not in allowed:
            raise ValueError(f"retrieval_strategy must be one of {allowed}, got '{v}'")
        return v

    @field_validator("default_llm")
    @classmethod
    def validate_default_llm(cls, v: str) -> str:
        allowed = {"llama3", "mistral"}
        if v not in allowed:
            raise ValueError(f"default_llm must be one of {allowed}, got '{v}'")
        return v

    @field_validator("embedding_device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        allowed = {"cpu", "cuda", "mps"}
        if v not in allowed:
            raise ValueError(f"embedding_device must be one of {allowed}, got '{v}'")
        return v

    def ensure_dirs(self) -> None:
        """Create runtime directories if they do not exist."""
        self.telemetry_store_dir.mkdir(parents=True, exist_ok=True)
        if self.dev_cache_llm_responses:
            self.dev_cache_dir.mkdir(parents=True, exist_ok=True)


# Singleton — import this everywhere.
settings = Settings()
