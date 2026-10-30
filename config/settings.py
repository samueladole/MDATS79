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

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


#: Canonical retrieval strategy names — the single source of truth.
#: ``pipeline.retrieval.RETRIEVERS`` asserts its keys match this at import time,
#: so the two cannot drift. Names must be single tokens: the experiment runner's
#: ``--conditions`` parser splits labels on '_' and expects exactly two parts.
RETRIEVAL_STRATEGIES: tuple[str, ...] = ("dense", "hybrid", "graph")


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
        default="http://host.docker.internal:11434",
        description="Base URL for the Ollama inference server.",
    )
    ollama_timeout_seconds: int = Field(
        default=120,
        description="Seconds to wait for a single LLM generation response.",
    )
    default_llm: str = Field(
        default="llama3",
        description="Default generator model for interactive queries. Must be in GENERATOR_MODELS.",
    )
    generator_models: str = Field(
        default="llama3,mistral",
        description=(
            "Comma-separated Ollama model tags offered as generators in the dashboard "
            "and CLI. llama3 and mistral are the dissertation models; any other tag "
            "(e.g. llama3.2:3b) works with the generic prompt."
        ),
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
        description="Weight given to dense scores in hybrid retrieval (0-1).",
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
    chroma_mode: str = Field(
        default="http",
        description=(
            "'embedded' runs ChromaDB inside the application, storing the index "
            "under CHROMA_PATH, so no database server is needed. 'http' connects "
            "to a ChromaDB server at CHROMA_HOST:CHROMA_PORT (the Docker setup)."
        ),
    )
    chroma_path: Path = Field(
        default=Path("data/chroma_store"),
        description="Index directory used when CHROMA_MODE=embedded.",
    )
    chroma_host: str = Field(default="localhost")
    chroma_port: int = Field(default=8000)
    chroma_collection: str = Field(default="ragscope_corpus")
    documents_collection: str = Field(
        default="ragscope_documents",
        description="Collection for your own PDF and text files, kept apart from BioASQ.",
    )
    documents_dir: Path = Field(
        default=Path("data/documents"),
        description="Folder of PDF, text and Markdown files ingested by --corpus documents.",
    )

    # ── RAGAS ──────────────────────────────────────────────────────────────────
    ragas_judge_model: str = Field(
        default="qwen2.5:7b",
        description="LLM used as the RAGAS judge for metric computation.",
    )
    ragas_judge_temperature: float = Field(
        default=0.0,
        description="Temperature for the RAGAS judge LLM (0-1).",
    )
    ragas_max_tokens: int = Field(default=2048)
    ragas_timeout_seconds: int = Field(
        default=180,
        description=(
            "Seconds each RAGAS judge call may take, including time queued behind "
            "the other metrics. 180 is RAGAS's own default; raise it on CPU-only machines."
        ),
    )

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

    # ── Graph retrieval ───────────────────────────────────────────────────────
    graph_path: Path = Field(
        default=Path("data/graph/graph.json"),
        description="NetworkX node-link JSON produced by the offline graph builder.",
    )
    graph_node_chunks_path: Path = Field(
        default=Path("data/graph/node_to_chunks.json"),
        description="{node_id: [chunk_id, ...]} provenance index for the graph.",
    )

    graph_fuse_dense: bool = Field(
        default=True,
        description=(
            "Fuse graph retrieval with the dense leg. Set False to return "
            "graph-only results — useful while graph coverage is below the "
            "dense index's, where fusion cannot surface graph-only chunks."
        ),
    )

    @field_validator("retrieval_strategy")
    @classmethod
    def validate_retrieval_strategy(cls, v: str) -> str:
        if v not in RETRIEVAL_STRATEGIES:
            raise ValueError(
                f"retrieval_strategy must be one of {set(RETRIEVAL_STRATEGIES)}, got '{v}'"
            )
        return v

    @property
    def generator_model_list(self) -> list[str]:
        """GENERATOR_MODELS as a list, in the order given."""
        return [m.strip() for m in self.generator_models.split(",") if m.strip()]

    @model_validator(mode="after")
    def validate_default_llm(self) -> Settings:
        models = self.generator_model_list
        if not models:
            raise ValueError("GENERATOR_MODELS must name at least one model")
        if self.default_llm not in models:
            raise ValueError(
                f"DEFAULT_LLM '{self.default_llm}' is not in GENERATOR_MODELS {models}"
            )
        return self

    @field_validator("chroma_mode")
    @classmethod
    def validate_chroma_mode(cls, v: str) -> str:
        allowed = {"embedded", "http"}
        if v not in allowed:
            raise ValueError(f"chroma_mode must be one of {allowed}, got '{v}'")
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
