# RAGScope — System Architecture

> **MSc Data Science Dissertation · Leeds Beckett University · 2026**  
> Design Science Research Artefact — Phase 2 Deliverable

---

## Table of Contents

- [Overview](#overview)
- [High-Level Architecture](#high-level-architecture)
- [Service Layer (Docker)](#service-layer-docker)
- [Module Dependency Map](#module-dependency-map)
- [Pipeline Stages](#pipeline-stages)
  - [Stage 0 — Configuration](#stage-0--configuration)
  - [Stage 1 — Data Ingestion](#stage-1--data-ingestion)
  - [Stage 2 — Query Execution](#stage-2--query-execution)
  - [Stage 3 — Telemetry](#stage-3--telemetry)
  - [Stage 4 — Evaluation](#stage-4--evaluation)
  - [Stage 5 — Dashboard](#stage-5--dashboard)
  - [Stage 6 — Experiments](#stage-6--experiments)
- [Data Flow Diagrams](#data-flow-diagrams)
  - [Ingestion Data Flow](#ingestion-data-flow)
  - [Query Execution Data Flow](#query-execution-data-flow)
- [Component Reference](#component-reference)
  - [config/settings.py](#configsettingspy)
  - [pipeline/embeddings.py](#pipelineembeddingspy)
  - [pipeline/vectorstore.py](#pipelinevectorstoropy)
  - [pipeline/retrieval/dense.py](#pipelineretrievaldensepy)
  - [pipeline/retrieval/hybrid.py](#pipelineretrievalhybridpy)
  - [pipeline/generation/llm\_client.py](#pipelinegenerationllm_clientpy)
  - [pipeline/generation/llama3.py and mistral.py](#pipelinegenerationllama3py-and-mistralpy)
  - [pipeline/ingestion.py](#pipelineingestionpy)
  - [pipeline/rag.py](#pipelineragpy)
  - [telemetry/timer.py](#telemetrytimerpypy)
  - [telemetry/token\_counter.py](#telemetrytoken_counterpy)
  - [telemetry/logger.py](#telemetryloggerpy)
  - [evaluation/ragas\_runner.py](#evaluationragas_runnerpy)
  - [evaluation/hallucination\_score.py](#evaluationhallucination_scorepy)
  - [evaluation/metrics.py](#evaluationmetricspy)
  - [evaluation/benchmark.py](#evaluationbenchmarkpy)
  - [dashboard/app.py and pages/](#dashboardapppy-and-pages)
  - [experiments/run\_2x2\_factorial.py](#experimentsrun_2x2_factorialpy)
  - [experiments/analyse\_results.py](#experimentsanalyse_resultspy)
- [Design Principles](#design-principles)
- [Experimental Conditions](#experimental-conditions)
- [Configuration Reference](#configuration-reference)
- [Key References](#key-references)

---

## Overview

RAGScope is a modular, framework-agnostic observability and evaluation platform for Retrieval-Augmented Generation (RAG) systems. The platform was designed and implemented following the **Design Science Research Methodology** (Peffers et al., 2007) as the primary research artefact for an MSc Data Science dissertation investigating RQ1–RQ3.

The system integrates five concerns into a single cohesive platform:

1. **Data preparation** — three benchmark corpora loaded, cleaned, chunked, embedded, and indexed
2. **RAG query execution** — two LLMs × two retrieval strategies, fully instrumented
3. **Real-time telemetry** — per-stage latency, token counts, and cost at every query
4. **Automated evaluation** — RAGAS metrics and composite hallucination risk scoring
5. **Observability interface** — interactive Streamlit dashboard and experiment analysis tooling

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            RAGScope Platform                                 │
│                                                                             │
│  ┌──────────────────┐   ┌──────────────────┐   ┌───────────────────────┐  │
│  │  Data Ingestion  │   │  Query Pipeline  │   │  Evaluation Layer     │  │
│  │                  │   │                  │   │                       │  │
│  │  MS MARCO        │   │  Query           │   │  RAGAS Runner         │  │
│  │  Natural Qs      │──▶│  ├─ Dense Ret.   │──▶│  ├─ Context Rel.     │  │
│  │  HotpotQA        │   │  └─ Hybrid Ret.  │   │  ├─ Faithfulness     │  │
│  │                  │   │                  │   │  ├─ Correctness       │  │
│  │  Clean → Chunk   │   │  LLM Generation  │   │  └─ Hallucin. Risk   │  │
│  │  Embed → Upsert  │   │  ├─ Llama 3      │   │                       │  │
│  └──────────────────┘   │  └─ Mistral 7B   │   └───────────┬───────────┘  │
│                         └────────┬─────────┘               │              │
│                                  │                          │              │
│  ┌───────────────────────────────▼──────────────────────────▼──────────┐  │
│  │                       Telemetry Layer                                │  │
│  │   Timer │ TokenCounter │ TelemetryLogger → JSON Store               │  │
│  └──────────────────────────────────┬───────────────────────────────── ┘  │
│                                     │                                      │
│  ┌──────────────────────────────────▼───────────────────────────────────┐  │
│  │              Streamlit Observability Dashboard                        │  │
│  │  Live Monitor │ Comparison │ Query Explorer │ Benchmark Results │     │  │
│  │  Knowledge Base                                                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Service Layer (Docker)

The platform is composed of two production Docker services orchestrated via `docker-compose.yml`, plus one development-only service defined in `docker-compose.override.yml`:

| Service | Image | Port | Responsibility |
|---|---|:---:|---|
| `chromadb` | `chromadb/chroma:latest` | 8000 | Persistent vector store (embedding storage and ANN search) |
| `ragscope` | `docker/app/Dockerfile` (target `app`) | 8501 | Main application (pipeline + evaluation + dashboard) |
| `jupyter` *(dev only)* | `docker/app/Dockerfile` (target `jupyter`) | 8888 | Notebook server for exploratory analysis; only started via the override file |

`docker/app/Dockerfile` is a multi-stage build with two runtime targets sharing one `builder` stage: `app` (production — no notebook dependencies) and `jupyter` (a separate `builder-notebooks` stage syncs the `notebooks` extra, then the `jupyter` runtime stage copies that venv instead of the lean one). Both `docker-compose.yml`'s `ragscope` service and `docker-compose.override.yml`'s `jupyter` service pin an explicit `target:` — Docker builds the *last* stage in the file when no target is given, so once a second runtime stage was added after `app`, leaving `target:` unset on the `ragscope` service would have silently built the wrong (larger, notebook-bundled) stage as the production image. `jupyterlab`/`matplotlib`/`seaborn`/`ipywidgets` are therefore baked in at **build** time for the `jupyter` target — cached like any other Docker layer, so `docker compose up` never re-downloads them — rather than `pip install`-ed at container start, which was tried first but re-ran the full install on every container start since nothing cached it.

Ollama is deliberately **not** a Docker service. It runs on the host machine — the operator pulls the required models there once (`ollama pull llama3`, `ollama pull mistral`) before starting the stack — and the `ragscope` (and `jupyter`) containers reach it at `http://host.docker.internal:11434` via the `extra_hosts: host.docker.internal:host-gateway` entry in the compose file. This keeps LLM inference on host compute without an extra virtualisation layer, at the cost of the host needing Ollama installed and running independently of `docker compose up`.

Services communicate over an internal bridge network (`ragscope_net`). The `ragscope` service depends on `chromadb` reaching a healthy state before starting, enforced by a Docker Compose `condition: service_healthy` dependency declaration. It does **not** wait on an Ollama container (there isn't one) — instead, `docker/scripts/entrypoint.sh` polls the host's Ollama HTTP API directly and fails fast with a diagnostic message if no model is available there.

**Startup order:**

```
chromadb (healthy) ──▶ ragscope entrypoint polls host Ollama for a pulled model ──▶ ragscope (starts dashboard)
```

---

## Module Dependency Map

```
config/settings.py
    │
    ├── data/preprocessing/cleaner.py
    ├── data/preprocessing/chunker.py
    │       │
    │       └── data/loaders/{msmarco, natural_questions, hotpotqa}.py
    │
    ├── pipeline/embeddings.py
    ├── pipeline/vectorstore.py
    │
    ├── pipeline/retrieval/dense.py ──────┐
    ├── pipeline/retrieval/hybrid.py ─────┤
    │                                     │
    ├── pipeline/generation/llm_client.py │
    ├── pipeline/generation/llama3.py ────┤
    ├── pipeline/generation/mistral.py ───┤
    │                                     │
    ├── pipeline/ingestion.py             │
    │                                     │
    ├── telemetry/timer.py                │
    ├── telemetry/token_counter.py        │
    ├── telemetry/logger.py               │
    │                                     │
    ├── evaluation/ragas_runner.py        │
    ├── evaluation/hallucination_score.py │
    ├── evaluation/metrics.py             │
    ├── evaluation/benchmark.py           │
    │                                     │
    └── pipeline/rag.py ◀─────────────────┘  (central orchestrator)
            │
            ├── dashboard/app.py
            │   ├── dashboard/pages/01_live_monitor.py
            │   ├── dashboard/pages/02_comparison.py
            │   ├── dashboard/pages/03_query_explorer.py
            │   ├── dashboard/pages/04_benchmark_results.py
            │   └── dashboard/pages/05_knowledge_base.py
            │
            └── experiments/run_2x2_factorial.py
                    └── experiments/analyse_results.py
```

---

## Pipeline Stages

### Stage 0 — Configuration

**Module:** `config/settings.py`

All runtime configuration is centralised in a single Pydantic `Settings` class loaded from the `.env` file. No module reads `os.environ` directly — all configuration flows through the `settings` singleton.

```python
from config.settings import settings

print(settings.ollama_base_url)    # http://host.docker.internal:11434
print(settings.top_k)              # 5
print(settings.retrieval_strategy) # "hybrid"
```

The settings class validates types and raises clear errors at startup on misconfiguration, preventing silent failures deep in the pipeline.

---

### Stage 1 — Data Ingestion

**Entry point:** `pipeline/ingestion.py` → `ingest_corpus()`  
**CLI:** `uv run python pipeline/ingestion.py --corpus all`

Ingestion is a streaming pipeline that processes documents in mini-batches to keep peak RAM consumption bounded regardless of corpus size.

**Substeps:**

| Step | Module | Description |
|---|---|---|
| Load | `data/loaders/{msmarco,nq,hotpotqa}.py` | Download and load raw passages from HuggingFace Datasets |
| Deduplicate | `pipeline/ingestion.py` | Track seen passage IDs in a set; skip cross-dataset duplicates |
| Clean | `data/preprocessing/cleaner.py` | Strip HTML, normalise unicode, collapse whitespace |
| Chunk | `data/preprocessing/chunker.py` | Token-bounded overlapping chunks via tiktoken |
| Embed | `pipeline/embeddings.py` | Dense vectors using `all-MiniLM-L6-v2` (384d, L2-normalised) |
| Upsert | `pipeline/vectorstore.py` | ChromaDB HTTP upsert (idempotent; cosine distance collection) |
| BM25 index | `pipeline/ingestion.py` | JSONL corpus persisted to `data/bm25_corpus.jsonl` |
| Manifest | `pipeline/ingestion.py` | `data/ingestion_manifest.json` written for Phase 3 reporting |

**Memory profile:** peak ≈ `embed_batch_size × avg_chunk_bytes + embed_batch_size × 384 × 4 bytes`. Default `embed_batch_size=256` ≈ 2.4 MB — safe on any consumer machine.

---

### Stage 2 — Query Execution

**Entry point:** `pipeline/rag.py` → `RAGPipeline.query()`

The `RAGPipeline` class is the single entry point for all query execution — used by the CLI, the dashboard, and the experiment runner.

**Execution sequence:**

```
user query
    │
    ▼
[1] Retrieve top-k chunks
    ├── dense:  embed query → ChromaDB ANN search
    └── hybrid: dense + BM25 → Reciprocal Rank Fusion
    │
    ▼
[2] Build prompt
    └── query + retrieved chunk texts injected via model-specific template
    │
    ▼
[3] LLM generation (Ollama local inference)
    ├── Llama 3 (8B) — [INST] chat template
    └── Mistral 7B  — [INST] / [/INST] chat template
    │
    ▼
[4] Token counting + cost estimation
    │
    ▼
[5] RAGAS evaluation
    ├── context_relevance
    ├── answer_faithfulness
    └── answer_correctness (if ground truth supplied)
    │
    ▼
[6] Hallucination risk scoring
    └── 1 − (0.6 × faithfulness + 0.4 × context_relevance)
    │
    ▼
[7] Telemetry record built and written to JSON store
    │
    ▼
RAGResult returned to caller
```

---

### Stage 3 — Telemetry

**Modules:** `telemetry/timer.py`, `telemetry/token_counter.py`, `telemetry/logger.py`

Every query execution is fully instrumented. Telemetry is collected **inline** with query execution — not in a separate thread or process — so timing measurements are accurate and there is no asynchronous complexity.

**`Timer`** is a context manager that records wall-clock elapsed time in milliseconds:

```python
with Timer("retrieval") as t:
    chunks = retriever.retrieve(query)
# t.elapsed_ms is now populated
```

**`TokenCounter`** uses tiktoken (`cl100k_base`) to count prompt and completion tokens and computes cost estimates from the configurable per-token rate table.

**`TelemetryLogger`** serialises a complete flat record dict to an individual `.json` file in `telemetry/store/`. The flat schema is designed for direct ingestion into `pandas.DataFrame` during analysis.

**Telemetry record schema** (selected fields):

| Field | Type | Description |
|---|---|---|
| `record_id` | str | UUID4 stable identifier |
| `timestamp_utc` | str | ISO-8601 UTC timestamp |
| `query` | str | Raw query text |
| `dataset` | str | Source benchmark dataset |
| `llm_model` | str | `llama3` or `mistral` |
| `retrieval_strategy` | str | `dense` or `hybrid` |
| `embed_query_ms` | float | Query embedding time |
| `vector_search_ms` | float | ChromaDB ANN search time (dense retrieval only; 0 for hybrid) |
| `dense_search_ms` | float | Dense candidate search time (hybrid retrieval only; 0 for dense) |
| `bm25_search_ms` | float | BM25 scoring time (hybrid retrieval only; 0 for dense) |
| `rrf_fusion_ms` | float | Reciprocal Rank Fusion merge time (hybrid retrieval only; 0 for dense) |
| `retrieval_ms` | float | Total retrieval wall-clock time (sum of the applicable sub-stages above) |
| `generation_ms` | float | LLM generation wall-clock time |
| `evaluation_ms` | float | RAGAS evaluation wall-clock time (0 if evaluation was skipped) |
| `e2e_ms` | float | End-to-end pipeline latency — retrieval + generation + evaluation, taken directly from the `Timer` wrapping the whole of `RAGPipeline.query()`, not recomputed from the other fields |
| `prompt_tokens` | int | Tokens in the full prompt |
| `completion_tokens` | int | Tokens in the generated response |
| `estimated_cost_usd` | float | Cost estimate in USD |
| `context_relevance` | float | RAGAS context relevance score |
| `answer_faithfulness` | float | RAGAS answer faithfulness score |
| `answer_correctness` | float | RAGAS answer correctness score |
| `hallucination_risk` | float | Composite risk score [0, 1] |
| `retrieved_chunks` | list | Top-k chunk IDs, texts (truncated), and scores |

> **Note:** `e2e_ms` includes RAGAS evaluation time, which can dominate total latency (the judge LLM call(s) routinely take several seconds to tens of seconds — often longer than retrieval and generation combined). Records written before this was corrected only sum `retrieval_ms + generation_ms` and will understate `e2e_ms` for queries that ran evaluation; the dashboard's Query Explorer flags this for older records where it's detectable.

---

### Stage 4 — Evaluation

**Modules:** `evaluation/ragas_runner.py`, `evaluation/hallucination_score.py`, `evaluation/metrics.py`

**RAGAS metrics** are computed using the `RAGASRunner`, which wraps the RAGAS ≥ 0.2 `EvaluationDataset` API. The auxiliary LLM judge (`settings.ragas_judge_model`, default: `qwen2.5:7b`) is held constant across all four experimental conditions to ensure metric comparability — this is a key methodological control.

The judge LLM is routed through local Ollama via `langchain_ollama.ChatOllama`, rebuilt fresh on every `evaluate()` call rather than cached on the runner. `ChatOllama` opens its async HTTP client once at construction, but `ragas.evaluate()` runs each call inside its own `asyncio.run(...)` (a new event loop every time); a cached client's connections stay bound to whichever loop first used them, so a shared instance across calls fails with `RuntimeError: Event loop is closed` on the second and every subsequent call. Building a new `ChatOllama` per call keeps it bound to the loop that will actually use it.

Embeddings for `AnswerCorrectness` (semantic similarity to the ground truth) reuse the pipeline's own `all-MiniLM-L6-v2` `EmbeddingGenerator` (`pipeline/embeddings.py`) via a small LangChain `Embeddings` adapter, rather than `langchain_ollama.OllamaEmbeddings` against the judge model — the judge is a chat model (`qwen2.5:7b`), and Ollama's embeddings endpoint rejects a model that wasn't loaded in embedding-serving mode. This adapter is synchronous and holds no event-loop-bound resources, so unlike the judge LLM it's safe to build once and reuse across calls. No external API calls are made anywhere in the evaluation path.

**Statistical analysis** (`evaluation/metrics.py`) is implemented in pure Python (no numpy dependency) to keep the module testable without the full ML stack:

- Descriptive statistics using linear-interpolation percentiles
- Cohen's d with pooled standard deviation
- Pearson r with pairwise missing-value exclusion

---

### Stage 5 — Dashboard

**Entry point:** `dashboard/app.py`  
**URL:** `http://localhost:8501`

The dashboard is a five-page Streamlit application providing real-time and retrospective views over the telemetry store and the ChromaDB collection.

| Page | File | Purpose |
|---|---|---|
| Live Monitor | `01_live_monitor.py` | Submit queries interactively; results persist across reruns (e.g. expanding a chunk) instead of gating on button state; per-session (not global) latency history; per-query pipeline flow Sankey |
| Comparison | `02_comparison.py` | RAGAS heatmaps and Cohen's d table across all four conditions; flags missing/sparse conditions instead of silently reporting a false zero effect size |
| Query Explorer | `03_query_explorer.py` | Per-query drill-down: chunks, scores, latency decomposition, pipeline flow Sankey, raw JSON |
| Benchmark Results | `04_benchmark_results.py` | Load benchmark CSVs; grouped bar of mean RAGAS scores, box plots, hallucination risk band distribution, Cohen's d effect-size chart, latency/token breakdown, Pearson correlation heatmap, per-dataset breakdown |
| Knowledge Base | `05_knowledge_base.py` | Read-only ChromaDB browser: collection stats, corpus composition by dataset, a 3D PCA projection of sampled embeddings, paginated chunk browser, semantic search preview |

Shared visualisation components live in `dashboard/components/`:

| Component | Description |
|---|---|
| `metric_cards.py` | Modern colour-graded RAGAS scorecard tiles (icon, value, meter bar, quality badge); `similarity_meter_html()` for chunk similarity bars; 6-column telemetry row (adds Evaluation Latency) |
| `latency_chart.py` | Time-series, histogram, grouped box-plot, and `pipeline_flow_sankey()` — a theme-aware Sankey diagram of one query's full stage-by-stage latency breakdown (Plotly) |
| `heatmap.py` | RAGAS score heatmap; token usage heatmap (Plotly) |
| `vector_space.py` | `pca_3d()` — pure-numpy PCA (via SVD, no scikit-learn/UMAP dependency) projecting sampled embeddings to 3D; `vector_space_scatter()` — Plotly 3D scatter coloured by dataset, for the Knowledge Base page |

**Colour conventions used throughout the dashboard:**
- Experimental conditions (Llama3/Mistral × Dense/Hybrid) use a fixed categorical colour per condition, never reassigned by row order.
- Continuous scores (chunk similarity, RAGAS scorecard tiles) use a red→amber→green gradient — green high / red low, inverted for Hallucination Risk where lower is better. Colour lives only on marks (bars, badges) and never on plain text, since a data colour used as text colour can be illegible depending on value and theme.
- Hallucination risk bands (LOW/MEDIUM/HIGH) use the fixed banding from `evaluation/hallucination_score.py` everywhere they appear, rather than a second, differently-thresholded scheme.
- The pipeline flow Sankey colours nodes by *phase*, not per-stage: every retrieval sub-stage (embedding, dense/BM25 search, RRF fusion) is a shade of blue, LLM generation is amber, RAGAS evaluation is red — so retrieval and generation are unmistakable at a glance rather than an arbitrary per-stage rainbow. Each node's label also carries a small RETRIEVAL/GENERATION/EVALUATION caption, and a colour-swatch legend renders below the chart.

---

### Stage 6 — Experiments

**Entry point:** `experiments/run_2x2_factorial.py`

The experiment runner executes all four conditions of the 2×2 factorial design against the 200-query benchmark. Each condition is processed sequentially. Results are written to a timestamped CSV in `experiments/results/`.

**Checkpoint/resume:** after each query, the completed query ID is appended to a checkpoint JSON file. If the run is interrupted, `--resume` re-reads the checkpoint and skips completed queries.

**Analysis:** `experiments/analyse_results.py` loads the benchmark CSV and generates a structured Markdown report with descriptive stats tables, pairwise Cohen's d, and Pearson correlations. The report is the primary quantitative output for dissertation Chapter 4.

---

## Data Flow Diagrams

### Ingestion Data Flow

```
HuggingFace Hub / Local Cache
          │
          ▼
  data/loaders/*.py          ← download + load raw passages
          │
          ▼
  data/preprocessing/
    cleaner.py               ← HTML strip, unicode normalise, whitespace collapse
    chunker.py               ← token-bounded overlapping chunks (tiktoken cl100k)
          │
          ├────────────────────────────────────────────────┐
          ▼                                                ▼
  pipeline/embeddings.py                      data/bm25_corpus.jsonl
  (SentenceTransformer                        (written incrementally
   all-MiniLM-L6-v2, 384d)                    via orjson JSONL)
          │
          ▼
  pipeline/vectorstore.py
  ChromaDB HTTP upsert
  (cosine distance, HNSW index)
          │
          ▼
  data/ingestion_manifest.json
  (Phase 3 deliverable)
```

### Query Execution Data Flow

```
User Query (str)
      │
      ├─────────────────────────────────────┐
      │  Dense path                         │  Hybrid path
      ▼                                     ▼
pipeline/embeddings.py          pipeline/embeddings.py
  embed_query() → [384d]          embed_query() → [384d]
      │                                     │
      ▼                                     ├──▶ ChromaDB ANN search (top-50)
pipeline/vectorstore.py                     │
  query() → RetrievedChunk[]               ├──▶ BM25Okapi.get_scores() (top-50)
                                            │
                                            ▼
                                       Reciprocal Rank Fusion
                                       rrf(r) = 1/(60 + rank)
                                       → merged RetrievedChunk[]
      │                                     │
      └─────────────────────────────────────┘
                       │
                       ▼
           pipeline/generation/{llama3,mistral}.py
           prompt template + Ollama /api/generate
                       │
                       ▼
           GenerationResponse
           {text, prompt_tokens, completion_tokens, generation_ms}
                       │
              ┌────────┴─────────┐
              ▼                  ▼
    telemetry/              evaluation/
    token_counter.py        ragas_runner.py
    timer.py                hallucination_score.py
              │                  │
              └────────┬─────────┘
                       ▼
           telemetry/logger.py
           build_record() → {record_id}.json
                       │
                       ▼
               RAGResult returned
```

---

## Component Reference

### `config/settings.py`

| Symbol | Type | Description |
|---|---|---|
| `Settings` | `BaseSettings` | Pydantic settings class; reads from `.env` |
| `settings` | `Settings` | Module-level singleton; import this everywhere |
| `settings.ensure_dirs()` | method | Creates `telemetry_store_dir` and cache dirs if absent |
| `settings.chroma_url` | property | Derived `http://{host}:{port}` string |

---

### `pipeline/embeddings.py`

| Symbol | Type | Description |
|---|---|---|
| `EmbeddingGenerator` | class | SentenceTransformer wrapper |
| `.embed(text)` | method | Embed one string → `list[float]` (384d, L2-normalised) |
| `.embed_batch(texts)` | method | Vectorised batch embedding with progress bar for >500 inputs |
| `.embed_query(query)` | method | Alias for `embed()` — semantic distinction for clarity |
| `get_embedding_generator()` | function | Returns module-level singleton; initialises on first call |

**Model:** `all-MiniLM-L6-v2` — 384-dimensional, compatible with ChromaDB cosine distance, fast on CPU.

---

### `pipeline/vectorstore.py`

| Symbol | Type | Description |
|---|---|---|
| `VectorStore` | class | ChromaDB HTTP client wrapper |
| `.add_chunks(chunks, embeddings)` | method | Batch upsert; idempotent by chunk_id |
| `.query(query_embedding, top_k)` | method | Cosine ANN search; returns `list[RetrievedChunk]` |
| `.count()` | method | Total chunks in the collection |
| `.count_where(where)` | method | Count chunks matching a metadata filter, without transferring documents/embeddings |
| `.get_chunks(where, limit, offset)` | method | Plain metadata-filtered browse (not similarity-ranked) — powers the Knowledge Base chunk browser; returned `RetrievedChunk.score` is always 0.0 |
| `.sample_embeddings(where, sample_size, seed)` | method | Random sample of chunks with `.embedding` populated, for the Knowledge Base's 3D vector-space view — fetches matching IDs first (cheap), samples in Python, then fetches full records only for the sampled IDs, since ChromaDB's `.get()` has no native random-sampling mode |
| `.embedding_dimension()` | method | Dimensionality of a stored embedding, read directly from the collection (`None` if empty) |
| `.reset()` | method | Delete and recreate collection (destructive) |
| `RetrievedChunk` | dataclass | `chunk_id`, `text`, `score` (cosine sim [0,1]), `metadata`, `embedding` (`list[float] \| None`, only populated by `sample_embeddings()`) |
| `get_vector_store()` | function | Module-level singleton |

**Distance conversion:** ChromaDB returns cosine distance ∈ [0, 2]. The wrapper converts to similarity via `max(0, 1 − distance/2)` so scores are always in [0, 1].

**Stale-collection recovery:** every method above is wrapped by a `_reconnect_on_stale_collection` decorator. `get_or_create_collection` caches a handle bound to the collection's UUID at connect time; if another process deletes and recreates the collection under the same name (e.g. `ingestion.py --reset`, or `VectorStore.reset()` from a different session) while this instance is alive, the server assigns a new UUID and the cached handle starts raising `NotFoundError` on every call even though a collection with that name exists again. The decorator catches this, re-fetches the collection by name, and retries once — transparent to the caller.

> A previous `list_documents()` method (a full-collection scan via a dummy zero-vector similarity query, reading a `metadata["source"]` field that nothing in the ingestion pipeline ever sets) has been removed — it was dead code backing a non-functional "browse documents" UI that has since been replaced by the Knowledge Base page's real, metadata-filtered browser above.

---

### `pipeline/retrieval/dense.py`

| Symbol | Type | Description |
|---|---|---|
| `DenseRetriever` | class | Pure vector similarity retrieval |
| `.retrieve(query, top_k)` | method | Embeds query; queries ChromaDB; returns `(chunks, telemetry_dict)` |

**Telemetry keys returned:** `strategy`, `top_k`, `embed_query_ms`, `vector_search_ms`, `retrieval_ms`, `chunks_retrieved`, `top_score`, `mean_score`.

---

### `pipeline/retrieval/hybrid.py`

| Symbol | Type | Description |
|---|---|---|
| `HybridRetriever` | class | BM25 + Dense + RRF fusion |
| `.build_bm25_index(ids, texts)` | method | Builds in-memory `BM25Okapi` index; call after loading BM25 corpus |
| `.retrieve(query, top_k)` | method | Full hybrid retrieval; returns `(chunks, telemetry_dict)` |
| `RRF_K` | constant | `60` — Cormack et al. (2009) smoothing constant |
| `_tokenise(text)` | function | Whitespace lowercased tokeniser for BM25 |

**RRF formula:**
```
rrf_score(chunk) = dense_weight × 1/(60 + dense_rank)
                 + bm25_weight  × 1/(60 + bm25_rank)
```

**Telemetry keys:** all dense keys plus `dense_search_ms`, `bm25_search_ms`, `rrf_fusion_ms`, `dense_candidates`, `bm25_candidates`, `dense_weight`, `bm25_weight`.

---

### `pipeline/generation/llm_client.py`

| Symbol | Type | Description |
|---|---|---|
| `BaseLLMClient` | ABC | Abstract base for all LLM clients |
| `.generate(query, context_chunks)` | method | POST to Ollama `/api/generate`; returns `GenerationResponse` |
| `GenerationResponse` | dataclass | `text`, `model`, `prompt_tokens`, `completion_tokens`, `generation_ms`, `raw` |
| `GenerationResponse.estimated_cost` | property | USD cost from configurable token rates |

**Retry policy:** 3 attempts with exponential backoff (2–10s) on `httpx.HTTPError` or `httpx.TimeoutException` via `tenacity`.

**System prompt:** instructs the model to answer only from retrieved context, discard prior knowledge, and express uncertainty if the context is insufficient.

---

### `pipeline/generation/llama3.py` and `mistral.py`

Each concrete client overrides `model_name` and `_build_prompt()` to apply the model's native chat template:

| Client | `model_name` | Chat template |
|---|---|---|
| `Llama3Client` | `llama3:8b` | `<\|begin_of_text\|>...<\|eot_id\|>` |
| `MistralClient` | `mistral:7b` | `<s>[INST] ... [/INST]` |

Applying the correct template is essential — supplying a completion-style prompt to an instruct-tuned model degrades faithfulness and can produce refusals.

---

### `pipeline/ingestion.py`

| Symbol | Type | Description |
|---|---|---|
| `ingest_corpus(corpus, ...)` | function | Full ingestion pipeline; returns `IngestionSummary` |
| `load_bm25_corpus()` | function | Streaming JSONL read → `(ids, texts)` for `HybridRetriever` |
| `IngestionSummary` | dataclass | Run statistics written to `data/ingestion_manifest.json` |
| `DatasetStats` | dataclass | Per-dataset counters (passages loaded, skipped, chunks created) |
| `BM25_CORPUS_PATH` | `Path` | `data/bm25_corpus.jsonl` |
| `MANIFEST_PATH` | `Path` | `data/ingestion_manifest.json` |

**CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--corpus` | `all` | `all` \| `msmarco` \| `natural_questions` \| `hotpotqa` |
| `--chunk-size` | `512` | Max tokens per chunk |
| `--chunk-overlap` | `64` | Token overlap between consecutive chunks |
| `--embed-batch-size` | `256` | Chunks per embedding forward pass (RAM control) |
| `--upsert-batch-size` | `500` | Chunks per ChromaDB upsert call |
| `--reset` | `False` | Wipe ChromaDB collection before ingesting |
| `--dry-run` | `False` | Estimate chunk counts without writing anything |

---

### `pipeline/rag.py`

| Symbol | Type | Description |
|---|---|---|
| `RAGPipeline` | class | End-to-end query orchestrator |
| `RAGPipeline.__init__(llm_model, retrieval_strategy, ...)` | method | Initialises retriever, LLM client, RAGAS runner |
| `RAGPipeline.query(query_text, ground_truth, ...)` | method | Full pipeline execution; returns `RAGResult` |
| `RAGResult` | dataclass | Complete query result including all scores and telemetry, incl. `evaluation_ms` |
| `RAGResult.summary()` | method | Formatted terminal output string (includes an Evaluation latency line) |

**`RAGPipeline` constructor behaviour:** if `retrieval_strategy == "hybrid"`, it loads `data/bm25_corpus.jsonl` via `load_bm25_corpus()` and calls `HybridRetriever.build_bm25_index()` immediately. This means the BM25 index build time is incurred once per pipeline instantiation, not per query.

**Timing scope:** `query()` wraps the entire method — retrieval, generation, token counting, *and* RAGAS evaluation — in a single `Timer("evaluation")`-nested-in-`Timer("e2e")` structure, and passes the resulting `e2e_timer.elapsed_ms` straight through to `build_record()` and `RAGResult.e2e_ms`. `evaluation_ms` is timed separately around the RAGAS call so it's visible on its own (e.g. in the dashboard's pipeline flow Sankey) rather than only implicit in the difference between `e2e_ms` and the other stages.

---

### `telemetry/timer.py`

```python
class Timer:
    """Context manager. Sets elapsed_ms on __exit__."""
    name: str
    elapsed_ms: float  # set to 0.0 until __exit__ is called
```

---

### `telemetry/token_counter.py`

| Symbol | Type | Description |
|---|---|---|
| `TokenCounter` | class | tiktoken-based counter (cl100k_base encoding) |
| `.build_usage(prompt_tokens, completion_tokens)` | method | Creates `TokenUsage` with cost estimate |
| `.from_generation_response(response)` | method | Builds `TokenUsage` from Ollama's reported token counts |
| `TokenUsage` | dataclass | `prompt_tokens`, `completion_tokens`, `total_tokens`, `estimated_cost_usd` |

> The live pipeline only ever counts tokens via `.from_generation_response()` — Ollama reports both prompt and completion token counts directly in its response, so nothing in `RAGPipeline.query()` re-tokenises text itself. `TokenCounter` previously also exposed `.count_tokens()`/`.count_prompt()` for re-tokenising raw text directly; those were removed as dead code (no caller in the live pipeline) — they existed only to support the one-off tiktoken cross-validation described in Chapter 3 §3.6.5, which can be re-run ad hoc against `tiktoken` directly if needed again.

---

### `telemetry/logger.py`

| Symbol | Type | Description |
|---|---|---|
| `build_record(...)` | function | Assembles a complete flat telemetry dict from all pipeline outputs |
| `TelemetryLogger` | class | Reads and writes individual JSON files in `telemetry/store/` |
| `.log(record)` | method | Serialises record via orjson; returns written `Path` |
| `.load_all(limit)` | method | Returns records sorted by mtime descending; used by dashboard |
| `.count()` | method | Number of JSON files in the store |
| `get_telemetry_logger()` | function | Module-level singleton |

**`build_record()` signature:** takes the *whole* `retrieval_telemetry` dict returned by `DenseRetriever`/`HybridRetriever.retrieve()` (not two cherry-picked fields), so every per-stage sub-timing it carries — `embed_query_ms`, and whichever of `vector_search_ms` or `dense_search_ms`/`bm25_search_ms`/`rrf_fusion_ms` applies — is persisted to the JSON record. It also takes `e2e_ms` and `evaluation_ms` directly from the caller's own timers rather than recomputing `e2e_ms` from `retrieval_ms + generation_ms` — the original implementation did this recomputation, which silently dropped RAGAS evaluation time from every persisted record (evaluation can be the dominant share of `e2e_ms`).

---

### `evaluation/ragas_runner.py`

| Symbol | Type | Description |
|---|---|---|
| `RAGASRunner` | class | RAGAS evaluation wrapper backed by local Ollama |
| `.evaluate(query, answer, contexts, ground_truth)` | method | Returns `dict` with three RAGAS metric scores |
| `get_ragas_runner()` | function | Module-level singleton |

**Judge model:** `settings.ragas_judge_model` (default `qwen2.5:7b`). Held constant across all four experimental conditions as a methodological control. RAGAS uses `langchain_ollama.ChatOllama` for LLM calls, rebuilt fresh per `evaluate()` call (see Stage 4 above for why — a cached client's async connections break across `ragas.evaluate()`'s per-call event loops). Embeddings for answer correctness use the pipeline's own `all-MiniLM-L6-v2` model via a small `Embeddings` adapter (`_SentenceTransformerEmbeddings`), not `OllamaEmbeddings` against the judge model — `qwen2.5:7b` is a chat model and Ollama's embeddings endpoint rejects it.

---

### `evaluation/hallucination_score.py`

| Symbol | Type | Description |
|---|---|---|
| `compute_hallucination_risk(faithfulness, context_relevance)` | function | Returns composite risk score or `None` if either input is `None` |
| `risk_band(score)` | function | `"LOW"` / `"MEDIUM"` / `"HIGH"` / `"UNKNOWN"` |
| `risk_colour(score)` | function | Hex colour for the risk band (`#2ecc71` / `#f39c12` / `#e74c3c` / `#95a5a6`) |
| `FAITHFULNESS_WEIGHT` | constant | `0.6` |
| `CONTEXT_RELEVANCE_WEIGHT` | constant | `0.4` |
| `LOW_THRESHOLD` | constant | `0.35` |
| `MEDIUM_THRESHOLD` | constant | `0.65` |

---

### `evaluation/metrics.py`

| Symbol | Type | Description |
|---|---|---|
| `descriptive_stats(values, metric)` | function | Returns `DescriptiveStats` (mean, std, median, IQR, min, max) |
| `cohens_d(group_a, group_b, ...)` | function | Returns `CohensD` with pooled SD and magnitude label |
| `pearson_r(x, y)` | function | Pearson correlation coefficient; `None` if < 3 paired values |
| `compare_conditions(records_a, records_b, ...)` | function | Full pairwise comparison dict for all `EVAL_METRICS` |
| `EVAL_METRICS` | constant | List of metric names used in analysis |

---

### `evaluation/benchmark.py`

| Symbol | Type | Description |
|---|---|---|
| `load_benchmark(seed)` | function | Returns shuffled 200-query list (60 MS MARCO + 40 NQ + 100 HotpotQA) |
| `run_benchmark(pipeline, queries, ...)` | function | Executes all queries; supports checkpoint/resume; returns `list[dict]` |

---

### `dashboard/app.py` and `pages/`

The Streamlit app uses multi-page routing via the `dashboard/pages/` directory. Pages are numbered to control sidebar order.

| Page | Route | Key imports |
|---|---|---|
| Home | `/` | `config/settings`, `telemetry/logger` |
| Live Monitor | `01_live_monitor` | `pipeline/rag.RAGPipeline`, all component modules incl. `latency_chart.pipeline_flow_sankey` |
| Comparison | `02_comparison` | `telemetry/logger`, `evaluation/metrics`, `dashboard/components/heatmap` |
| Query Explorer | `03_query_explorer` | `telemetry/logger`, `evaluation/hallucination_score`, all component modules incl. `latency_chart.pipeline_flow_sankey` |
| Benchmark Results | `04_benchmark_results` | `pandas`, `plotly.express`, `evaluation/metrics` |
| Knowledge Base | `05_knowledge_base` | `pipeline/vectorstore.get_vector_store`, `pipeline/embeddings.get_embedding_generator`, `dashboard/components/vector_space` |

---

### `experiments/run_2x2_factorial.py`

Typer CLI application. Resolves the conditions to run, loads the benchmark once, then iterates over conditions calling `RAGPipeline` + `run_benchmark` for each. Writes a timestamped CSV to `experiments/results/`.

**Conditions:**

| Condition | LLM | Retrieval |
|:---:|---|---|
| A | Llama 3 (8B) | Dense |
| B | Llama 3 (8B) | Hybrid |
| C | Mistral 7B | Dense |
| D | Mistral 7B | Hybrid |

---

### `experiments/analyse_results.py`

Typer CLI application. Reads a benchmark CSV and writes a Markdown report with four sections: descriptive statistics, pairwise Cohen's d, Pearson correlations, and per-dataset breakdown.

---

## Design Principles

**Single entry point per concern.** Every aspect of the system has one canonical entry point: `RAGPipeline.query()` for query execution, `ingest_corpus()` for data preparation, `TelemetryLogger.log()` for persistence. This makes the system easy to test and reason about.

**Singletons for expensive resources.** The embedding model, vector store client, RAGAS runner, and telemetry logger are all lazily initialised singletons (`get_*()` functions). This prevents redundant model loading and database connections across the pipeline. The one deliberate exception is `RAGASRunner`'s judge LLM: the `RAGASRunner` object itself is a singleton, but its `ChatOllama` client is rebuilt on every `evaluate()` call rather than cached, because RAGAS runs each call in its own asyncio event loop and a cached async client's connections don't survive across loops (see Stage 4 in the pipeline stages section).

**Flat telemetry schema.** Every telemetry record uses a flat dict rather than nested objects, simplifying `pandas.read_json()` ingestion and downstream analysis without transformation steps.

**Pure-Python statistics.** `evaluation/metrics.py` has no numpy or scipy dependency. Cohen's d and Pearson r are implemented from first principles using only the standard library. This makes the module testable without the full ML stack and documents the statistical formulae explicitly.

**Modularity over convenience.** Dense and hybrid retrievers, Llama 3 and Mistral clients, and all three dataset loaders are independent modules that share a consistent interface. Swapping a component requires changing one constructor argument, not rewriting pipeline logic.

---

## Experimental Conditions

The 2×2 factorial design crosses two independent variables:

```
                     ┌─────────────────┬─────────────────┐
                     │   Dense Ret.    │   Hybrid Ret.   │
        ┌────────────┼─────────────────┼─────────────────┤
        │ Llama 3    │  Condition A    │  Condition B    │
        ├────────────┼─────────────────┼─────────────────┤
        │ Mistral 7B │  Condition C    │  Condition D    │
        └────────────┴─────────────────┴─────────────────┘
```

Each condition is evaluated against all 200 benchmark queries, producing 800 total telemetry records. Statistical comparisons use Cohen's d to quantify practical significance beyond p-values, and Pearson r to characterise the latency–faithfulness trade-off.

---

## Configuration Reference

All settings live in `.env` (see `.env.example`). Key values:

| Variable | Default | Used by |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | All LLM clients, RAGAS runner |
| `DEFAULT_LLM` | `llama3` | `RAGPipeline` default |
| `RETRIEVAL_STRATEGY` | `hybrid` | `RAGPipeline` default |
| `TOP_K` | `5` | Both retrievers |
| `HYBRID_DENSE_WEIGHT` | `0.6` | `HybridRetriever` RRF |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | `EmbeddingGenerator`; also backs `RAGASRunner`'s answer-correctness embeddings |
| `CHROMA_HOST` | `chromadb` | `VectorStore` |
| `CHROMA_COLLECTION` | `ragscope_corpus` | `VectorStore` |
| `RAGAS_JUDGE_MODEL` | `qwen2.5:7b` | `RAGASRunner` |
| `TELEMETRY_STORE_DIR` | `./telemetry/store` | `TelemetryLogger` |
| `EXPERIMENT_RANDOM_SEED` | `42` | Dataset loaders, sampling |

---

## Key References

- Barnett et al. (2024) — Seven failure points when engineering a RAG system
- Cormack, G.V., Clarke, C.L.A. and Buettcher, S. (2009) — Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods, *SIGIR 2009*
- Es et al. (2023) — RAGAS: Automated Evaluation of Retrieval Augmented Generation
- Hevner et al. (2004) — Design Science in Information Systems Research, *MIS Quarterly* 28(1)
- Karpukhin et al. (2020) — Dense Passage Retrieval for Open-Domain Question Answering, *EMNLP 2020*
- Lewis et al. (2020) — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, *NeurIPS 2020*
- Peffers et al. (2007) — A Design Science Research Methodology for Information Systems Research, *JMIS* 24(3)
- Robertson and Zaragoza (2009) — The Probabilistic Relevance Framework: BM25 and Beyond