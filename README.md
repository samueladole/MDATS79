<div align="center">

# 🔭 RAGScope
### Real-Time Observability & Evaluation Platform for Retrieval-Augmented Generation Systems

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![uv](https://img.shields.io/badge/uv-package_manager-DE5FE9?style=flat-square&logo=astral&logoColor=white)](https://docs.astral.sh/uv)
[![Docker](https://img.shields.io/badge/Docker-compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4%2B-orange?style=flat-square)](https://trychroma.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local_Inference-black?style=flat-square)](https://ollama.com)
[![RAGAS](https://img.shields.io/badge/RAGAS-Evaluation_Framework-6c63ff?style=flat-square)](https://docs.ragas.io)
[![Licence: MIT](https://img.shields.io/badge/Licence-MIT-green?style=flat-square)](LICENSE)
[![MSc Dissertation](https://img.shields.io/badge/MSc-Data_Science-522D80?style=flat-square)](https://www.leedsbeckett.ac.uk)

---

*MSc Data Science Dissertation Project — School of Built Environment, Engineering and Computing*
*Leeds Beckett University · March 2026 – September 2026*

</div>

---

## Table of Contents

- [Overview](#overview)
- [Research Context](#research-context)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Datasets](#datasets)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Option A - Docker (Recommended)](#option-a--docker-recommended)
  - [Option B - Local with uv](#option-b--local-with-uv)
  - [Configuration](#configuration)
- [Usage](#usage)
  - [Running the RAG Pipeline](#running-the-rag-pipeline)
  - [Launching the Dashboard](#launching-the-dashboard)
  - [Running the Benchmark Evaluation](#running-the-benchmark-evaluation)
- [Docker Reference](#docker-reference)
- [Evaluation Metrics](#evaluation-metrics)
- [Experimental Design](#experimental-design)
- [Results](#results)
- [Limitations](#limitations)
- [Research Outputs](#research-outputs)
- [Citation](#citation)
- [Licence](#licence)
- [Acknowledgements](#acknowledgements)

---

## Overview

**RAGScope** is an open-source, research-grade observability and evaluation platform for Retrieval-Augmented Generation (RAG) systems. It provides real-time telemetry collection, multi-dimensional quality evaluation using the RAGAS framework, and an interactive diagnostic dashboard — all in a single, framework-agnostic Python toolkit.

RAG systems are powerful but opaque. When a response is wrong, it is rarely obvious *where* the pipeline failed: was it a poor query embedding, a retrieval miss, an irrelevant chunk surfaced to the top, or the language model hallucinating despite a good context? RAGScope instruments every stage of the pipeline to answer exactly that question, in real time.

This research is structured around three questions:

- **RQ1 — Metric Calibration:** To what extent can a composite score based on RAGAS-compatible evaluation metrics be calibrated against ground-truth correctness to capture hallucination likelihood and retrieval faithfulness in RAG systems operating under real-time operational constraints?
- **RQ2 — Fault Localisation:** To what extent does a platform that integrates real-time telemetry with RAGAS-compatible evaluation metrics localise the source of a RAG system failure at the retrieval or generation stage, using per-stage metric signals captured at query-level granularity?
- **RQ3 — Configuration Trade-offs:** What performance trade-offs exist between dense and hybrid retrieval strategies, and between different open-source LLM configurations, when measured jointly across hallucination risk, retrieval faithfulness, response latency, and token cost?

---

## Research Context

This platform was developed as the primary artefact for an MSc Data Science dissertation at Leeds Beckett University. The research follows the **Design Science Research Methodology** (Peffers et al., 2007), in which the construction and rigorous empirical evaluation of a novel software artefact constitutes a legitimate and substantive contribution to knowledge.

### The Problem

Despite the maturity of evaluation frameworks like RAGAS (Es et al., 2023), a critical gap remains:

| Tool | Offline Batch Eval | Real-Time Telemetry | Open Source | Framework-Agnostic | Research-Grade Metrics |
|---|:---:|:---:|:---:|:---:|:---:|
| **RAGAS** | ✅ | ❌ | ✅ | ✅ | ✅ |
| **LangSmith** | ✅ | ✅ | ❌ | ❌ | Partial |
| **TruLens** | ✅ | Partial | ✅ | Partial | Partial |
| **Arize Phoenix** | ✅ | ✅ | ✅ | Partial | ❌ |
| **RAGScope** *(this work)* | ✅ | ✅ | ✅ | ✅ | ✅ |

RAGScope fills the integration gap: it combines the evaluative richness of RAGAS with production-style real-time telemetry and interactive visualisation, without locking the user into a specific orchestration framework or commercial platform.

---

## System Architecture

The platform is organised into six stages, each mapped to a specific research question or SMART objective — see `docs/ragscope_system_architecture.svg` for the full diagram (and `docs/architecture.md` for the detailed prose description of every module).

```
 1. Data          2. RAG Query Pipeline        3. Evaluation
 ────────    ▶    ─────────────────────   ▶    ─────────────
 3 corpora        Dense/Hybrid retrieval        RAGAS Runner (judge: qwen2.5:7b)
 → preprocess     → LLM generation              → Hallucination Risk score
 → index          (Llama 3 8B / Mistral 7B)     (RQ1)
 (RQ3)            (RQ3)

                                                        │
                                                        ▼

 6. Experiments   ◀────────────────────────────  4. Telemetry
 ────────────                                    ─────────────
 4 conditions ×                                  Per-stage latency + tokens
 200 queries → CSV                               → flat JSON record (RQ2)
 → Cohen's d, Pearson r                                │
 (RQ3 / Objective 6)                                   ▼

                                                  5. Dashboard
                                                  ─────────────
                                                  Live Monitor · Comparison ·
                                                  Query Explorer · Benchmark
                                                  Results · Knowledge Base
                                                  (Objective 5)
```

---

## Features

### 🔬 Real-Time Evaluation
- Per-query RAGAS metric computation (context relevance, answer faithfulness, answer correctness)
- Composite hallucination risk scoring derived from faithfulness and relevance signals
- Evaluation runs alongside query execution — no separate batch job required

### 📡 Telemetry Collection
- End-to-end latency profiling with stage-level decomposition (retrieval time vs. generation time)
- Token usage logging (prompt tokens, completion tokens, total)
- Per-query cost estimation based on configurable per-token rate tables
- Structured JSON telemetry store for reproducibility and offline analysis

### 🔁 Comparative Experimentation
- Pluggable retrieval strategies: **dense retrieval** (ChromaDB vector search) and **hybrid retrieval** (BM25 + dense via Reciprocal Rank Fusion)
- Pluggable LLM backends: **Llama 3 (8B)** and **Mistral 7B** via Ollama (runs fully locally — no API key required)
- 2 × 2 factorial experiment runner with automated result logging

### 📊 Interactive Dashboard
- Live metric time series for active query sessions, with results persisted across page interaction
- Comparative performance heatmaps across retrieval × LLM conditions
- Retrieved document chunk viewer with colour-graded per-chunk similarity scores
- Sankey diagram of the full pipeline process flow — every instrumented stage (embedding, retrieval sub-stages, generation, RAGAS evaluation) sized by actual wall-clock duration for a single query
- Read-only ChromaDB knowledge-base browser with corpus composition breakdown and a semantic search preview
- Export of session data to CSV for downstream analysis

### 🗂️ Benchmark Pipeline
- Pre-built loaders for MS MARCO, Natural Questions, and HotpotQA
- Stratified query sampling utilities
- Ground-truth answer alignment for RAGAS answer correctness computation

---

## Datasets

Three publicly available benchmark datasets are used for evaluation. All are available under open licences permitting academic re-use.

### MS MARCO — Passage Ranking
- **Source:** [microsoft.github.io/msmarco](https://microsoft.github.io/msmarco/)
- **Licence:** MIT
- **Usage in this project:** 50,000 passages indexed as the primary retrieval corpus; 60 development-set queries (30 single-answer, 30 multi-passage) used for evaluation
- **Why:** Large-scale, web-sourced passage retrieval — tests the platform under realistic retrieval corpus conditions

### Natural Questions (NQ)
- **Source:** [ai.google.com/research/NaturalQuestions](https://ai.google.com/research/NaturalQuestions)
- **Licence:** CC BY-SA 3.0
- **Usage in this project:** 40 development-set queries with Wikipedia ground-truth answers
- **Why:** Real Google Search queries — naturalistic, unambiguous ground-truth answers enable precise faithfulness measurement

### HotpotQA
- **Source:** [hotpotqa.github.io](https://hotpotqa.github.io/)
- **Licence:** CC BY-SA 4.0
- **Usage in this project:** 100 development-set queries (50 bridge, 50 comparison) from the full-wiki setting
- **Why:** Multi-hop reasoning — the hardest condition for RAG faithfulness, and the most diagnostic test for hallucination detection

| Dataset | Query Type | Corpus Source | Queries Used | Licence |
|---|---|---|:---:|---|
| MS MARCO | Passage ranking | Web documents | 60 | MIT |
| Natural Questions | Single-hop QA | Wikipedia | 40 | CC BY-SA 3.0 |
| HotpotQA | Multi-hop QA | Wikipedia (full) | 100 | CC BY-SA 4.0 |
| **Total** | | | **200** | |

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.12 | Core pipeline and evaluation |
| Package Manager | [uv](https://docs.astral.sh/uv) | Dependency management, venv, lockfile (`uv.lock`) |
| Containerisation | [Docker](https://docker.com) + [Compose](https://docs.docker.com/compose/) | Reproducible, isolated service orchestration |
| LLM Inference | [Ollama](https://ollama.com) | Local Llama 3 & Mistral 7B serving — runs on the **host** machine, not as a Compose service (containers reach it via `host.docker.internal`) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Dense vector generation |
| Vector DB | [ChromaDB](https://trychroma.com) | Embedding storage & similarity search (runs as a Compose service) |
| Sparse Retrieval | `rank_bm25` | BM25 scoring for hybrid retrieval |
| Hybrid Fusion | Custom RRF implementation | Reciprocal Rank Fusion |
| Evaluation | [RAGAS](https://docs.ragas.io) | Context relevance, faithfulness, correctness (judge: Qwen2.5 7B) |
| Telemetry | Custom JSON logger | Query-level instrumentation store |
| Dashboard | [Streamlit](https://streamlit.io) | Interactive real-time visualisation |
| Data Processing | `pandas`, `numpy` | Benchmark loading and analysis |
| Visualisation | `plotly` | Dashboard charts, heatmaps, and Sankey process-flow diagrams |
| Statistics | `statsmodels` | OLS trendlines on dashboard scatter plots |

---

## Project Structure

```
ragscope/
│
├── README.md
├── pyproject.toml                 # Project metadata & dependencies (uv)
├── uv.lock                        # Pinned lockfile — commit this for reproducibility
├── .env.example                   # Environment variable template
├── .dockerignore
├── .gitignore
├── LICENSE
│
├── docker-compose.yml             # Orchestrates production services: app, chromadb (Ollama runs on the host, not a Compose service)
├── docker-compose.override.yml    # Local dev overrides (hot-reload, volume mounts) + dev-only jupyter service
│
├── docker/                        # Per-service Dockerfiles
│   ├── app/
│   │   └── Dockerfile             # RAGScope app image (Python 3.12, uv-installed deps); also reused by the dev jupyter service
│   └── scripts/
│       ├── entrypoint.sh          # Container startup: waits for host Ollama + ChromaDB to be reachable
│
├── data/                          # Dataset loading and preprocessing
│   ├── loaders/
│   │   ├── msmarco.py             # MS MARCO passage & query loader
│   │   ├── natural_questions.py   # NQ development set loader
│   │   └── hotpotqa.py            # HotpotQA full-wiki loader
│   ├── preprocessing/
│   │   ├── chunker.py             # Text chunking strategies
│   │   └── cleaner.py             # Text normalisation utilities
│   └── raw/                       # Downloaded datasets (git-ignored)
│
├── pipeline/                      # Core RAG pipeline
│   ├── ingestion.py               # Document ingestion orchestrator
│   ├── embeddings.py              # Embedding generation wrapper
│   ├── vectorstore.py             # ChromaDB interface
│   ├── retrieval/
│   │   ├── dense.py               # Dense retrieval (vector similarity)
│   │   └── hybrid.py              # Hybrid retrieval (BM25 + RRF)
│   ├── generation/
│   │   ├── llm_client.py          # Ollama API wrapper
│   │   ├── llama3.py              # Llama 3 (8B) configuration
│   │   └── mistral.py             # Mistral 7B configuration
│   └── rag.py                     # End-to-end query pipeline orchestrator
│
├── telemetry/                     # Instrumentation layer
│   ├── logger.py                  # Structured JSON telemetry logger
│   ├── timer.py                   # Stage-level latency profiling
│   ├── token_counter.py           # Token usage and cost estimation
│   └── store/                     # Telemetry output (git-ignored)
│
├── evaluation/                    # Evaluation framework
│   ├── ragas_runner.py            # RAGAS metric computation wrapper
│   ├── metrics.py                 # Metric aggregation and scoring
│   ├── hallucination_score.py     # Composite hallucination risk scorer
│   └── benchmark.py              # 200-query benchmark runner
│
├── dashboard/                     # Streamlit observability dashboard
│   ├── app.py                     # Main Streamlit application entry point
│   ├── pages/
│   │   ├── 01_live_monitor.py     # Real-time query stream view + pipeline flow Sankey
│   │   ├── 02_comparison.py       # Cross-condition comparative analysis
│   │   ├── 03_query_explorer.py   # Per-query drill-down view + pipeline flow Sankey
│   │   ├── 04_benchmark_results.py # Benchmark experiment results
│   │   └── 05_knowledge_base.py   # Read-only ChromaDB browser + semantic search preview
│   └── components/
│       ├── theme.py               # Shared icons (Material Symbols / inline SVG) + CSS theme
│       ├── metric_cards.py        # Metric display components
│       ├── latency_chart.py       # Latency visualisation
│       └── heatmap.py             # RAGAS score heatmaps
│
├── experiments/                   # Experimental scripts
│   ├── run_2x2_factorial.py       # Full 2×2 factorial experiment runner
│   ├── analyse_results.py         # Statistical analysis (Cohen's d, etc.)
│   └── results/                   # Experiment output CSVs (git-ignored)
│
├── notebooks/                     # Exploratory analysis notebooks
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_retrieval_strategy_comparison.ipynb
│   └── 03_hallucination_analysis.ipynb
│
├── tests/                         # Unit and integration tests
│   ├── test_pipeline.py
│   ├── test_retrieval.py
│   ├── test_evaluation.py
│   └── test_telemetry.py
│
└── docs/                          # Extended documentation
    ├── architecture.md                              # Full module-by-module architecture reference
    ├── datasets.md                                   # Dataset selection, sampling, and provenance
    ├── evaluation_metrics.md                         # RAGAS + hallucination risk metric definitions
    ├── ragscope_system_architecture.svg               # System architecture diagram (6 stages)
    ├── architecture_simplification_analysis.md        # Supervisor-feedback-driven simplification audit
    ├── architecture_simplification_plan.md            # Diagram redesign proposal
    └── architecture_simplification_changelog.md       # Record of what was actually changed
```

---

## Getting Started

### Prerequisites

| Requirement | Version | Install |
|---|---|---|
| [Docker](https://docs.docker.com/get-docker/) | ≥ 26.0 | Required for the recommended Docker path |
| [Docker Compose](https://docs.docker.com/compose/) | ≥ 2.27 | Bundled with Docker Desktop |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | ≥ 0.4 | Required for the local development path |
| [Ollama](https://ollama.com) | Latest | **Required for both paths** — Ollama always runs on the host, never inside a container. Install it and pull the models before starting either path (see below). |
| RAM | ≥ 16 GB | 32 GB recommended for running both models |
| Disk | ≥ 25 GB free | Models (~14 GB for all three) + corpus + embeddings + images |

> **macOS / Linux — install uv in one line:**
> ```bash
> curl -LsSf https://astral.sh/uv/install.sh | sh
> ```
> **Windows:**
> ```powershell
> powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
> ```

---

### Option A — Docker (Recommended)

Docker Compose orchestrates two services automatically: the RAGScope app and ChromaDB (vector store). No manual setup of Python or ChromaDB is required — but **Ollama is not containerised**: it always runs on the host machine (reached via `host.docker.internal`), so it must be installed and have its models pulled *before* starting the stack, for both this path and Option B.

**1. Clone the repository**

```bash
git clone https://github.com/samueladole/ragscope.git
cd ragscope
```

**2. Install Ollama and pull the required models (on the host)**

```bash
# Install from https://ollama.com, then:
ollama serve &                # Start the Ollama daemon in the background, if not already running

ollama pull llama3             # ~4.7 GB
ollama pull mistral            # ~4.4 GB
ollama pull qwen2.5:7b         # ~4.7 GB (RAGAS judge model)
```

The container entrypoint polls this host Ollama instance on startup and fails fast with a diagnostic message if no model is available — it does **not** pull models itself.

**3. Configure environment variables**

```bash
cp .env.example .env
# Edit .env if you need to change defaults (see Configuration below)
```

**4. Build and start the services**

```bash
docker compose up --build
```

The dashboard will be available at **http://localhost:8501** once both services are healthy.

```
[+] Running 2/2
 ✔ chromadb     Started   → http://localhost:8000
 ✔ ragscope     Started   → http://localhost:8501
```

**5. Ingest the benchmark datasets**

Run this once to download and embed the MS MARCO, NQ, and HotpotQA passages into ChromaDB:

```bash
docker compose exec ragscope uv run python pipeline/ingestion.py \
  --corpus all \
  --chunk-size 512 \
  --chunk-overlap 64
```

> This embeds ~50,000 passages using `all-MiniLM-L6-v2`. Expect 20–40 minutes on CPU. Progress is displayed in the terminal.

**6. Stop all services**

```bash
docker compose down          # Stops containers, preserves volumes
docker compose down -v       # Stops and removes all volumes (full reset)
```

---

### Option B — Local with uv

Use this path for active development or when you need to run notebooks interactively.

**1. Clone and enter the repository**

```bash
git clone https://github.com/samueladole/ragscope.git
cd ragscope
```

**2. Install dependencies with uv**

uv reads `pyproject.toml` and creates an isolated virtual environment automatically. There is no separate `venv` activation step required.

```bash
uv sync                      # Installs all dependencies from uv.lock
```

To add or update a dependency:

```bash
uv add <package>             # Add a new package (updates pyproject.toml + uv.lock)
uv add --dev pytest          # Add a dev-only dependency
uv lock --upgrade            # Upgrade all packages to latest compatible versions
```

**3. Install and start Ollama**

Install Ollama from [ollama.com](https://ollama.com), then:

```bash
ollama serve &               # Start the Ollama daemon in the background

ollama pull llama3           # ~4.7 GB
ollama pull mistral          # ~4.4 GB
ollama pull qwen2.5:7b       # ~4.7 GB (LLM Judge)
```

**4. Start ChromaDB**

```bash
uv run chroma run --path ./data/chroma_store --port 8000
```

**5. Download and ingest datasets**

```bash
uv run python pipeline/ingestion.py --corpus all --chunk-size 512 --chunk-overlap 64
```

**6. Run tests to verify setup**

```bash
uv run pytest tests/ -v
```

---

### Configuration

All configuration is controlled via a single `.env` file, which is loaded by both Docker Compose and the local uv runner.

```bash
cp .env.example .env
```

```env
# ── LLM ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL=http://host.docker.internal:11434   # Docker: host.docker.internal | Local (no Docker): localhost
DEFAULT_LLM=llama3                       # llama3 | mistral

# ── Retrieval ───────────────────────────────────────────────────────────────
RETRIEVAL_STRATEGY=hybrid             # dense | hybrid
TOP_K=5                               # Number of retrieved passages per query

# ── Embeddings ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL=all-MiniLM-L6-v2

# ── ChromaDB ────────────────────────────────────────────────────────────────
CHROMA_HOST=chromadb                  # Docker: service name; local: localhost
CHROMA_PORT=8000
CHROMA_COLLECTION=ragscope_corpus

# ── RAGAS ───────────────────────────────────────────────────────────────────
RAGAS_JUDGE_MODEL=qwen2.5:7b          # LLM used to compute RAGAS metrics

# ── Cost Estimation ─────────────────────────────────────────────────────────
COST_PER_1K_PROMPT_TOKENS=0.00        # 0.00 for local inference; set for cloud models
COST_PER_1K_COMPLETION_TOKENS=0.00

# ── Telemetry ───────────────────────────────────────────────────────────────
TELEMETRY_STORE_DIR=./telemetry/store
```

> **Docker vs. local hostnames:** `CHROMA_HOST` differs by path — the real Compose service name (`chromadb`) inside Docker, `localhost` when running locally. `OLLAMA_BASE_URL` differs the other way: `host.docker.internal` inside Docker (Ollama is never a Compose service — it's always on the host), `localhost` when running locally, since there's no container boundary to cross.

---

## Usage

All commands below are shown in two variants: **Docker** (recommended) and **local uv**. They are functionally identical.

### Running the RAG Pipeline

Run a single query through the full pipeline (retrieval + generation + telemetry + evaluation):

```bash
# Docker
docker compose exec ragscope uv run python pipeline/rag.py \
  "What are the main causes of hallucination in large language models?" \
  --llm llama3 \
  --retrieval hybrid \
  --top-k 5

# Local
uv run python pipeline/rag.py \
  "What are the main causes of hallucination in large language models?" \
  --llm llama3 \
  --retrieval hybrid \
  --top-k 5
```

> The query is a positional argument, not a `--query` flag — `pipeline/rag.py` is a Typer CLI with `query` declared as `typer.Argument`.

**Sample output:**

```
────────────────────────────────────────────────────
 Query: What are the main causes of hallucination...
────────────────────────────────────────────────────
 Retrieved Chunks (top 5, hybrid):
   [1] similarity=0.87 | "Hallucinations in LLMs arise from..."
   [2] similarity=0.83 | "Training data biases contribute to..."
   ...

 Answer: Hallucination in large language models primarily stems from...

 ── Telemetry ──────────────────────────────────────
  Retrieval latency:    0.34s
  Generation latency:   3.21s
  Evaluation latency:   4.87s
  Total latency:        8.42s
  Prompt tokens:        847
  Completion tokens:    312
  Estimated cost:       $0.000 (local inference)

 ── RAGAS Evaluation ───────────────────────────────
  Context Relevance:    0.81
  Answer Faithfulness:  0.74
  Answer Correctness:   0.69
  Hallucination Risk:   LOW  (0.26)
────────────────────────────────────────────────────
 Telemetry logged → telemetry/store/a3f9c2e1-...-9b7d.json
────────────────────────────────────────────────────
```

### Launching the Dashboard

```bash
# Docker — already running; visit:
open http://localhost:8501

# Local
uv run streamlit run dashboard/app.py
```

The dashboard provides five views at `http://localhost:8501`:

- **Live Monitor** — submit a query interactively and watch the full pipeline execute: generated answer, RAGAS scores, colour-graded chunk similarity, and a Sankey diagram of every instrumented stage's wall-clock duration
- **Comparison** — side-by-side heatmaps of RAGAS scores across LLM × retrieval conditions
- **Query Explorer** — click any logged query to inspect retrieved chunks, similarity scores, faithfulness breakdowns, latency decomposition, and the same per-query pipeline flow Sankey
- **Benchmark Results** — full visualisation of the 200-query benchmark experiment results
- **Knowledge Base** — read-only browser of the ChromaDB collection: corpus composition by dataset, a paginated chunk browser, and a semantic search preview

### Running the Benchmark Evaluation

The full 2 × 2 factorial benchmark (200 queries × 4 conditions = 800 records):

```bash
# Docker
docker compose exec ragscope uv run python experiments/run_2x2_factorial.py \
  --dataset all \
  --conditions all \
  --output experiments/results/benchmark_20260801.csv

# Local
uv run python experiments/run_2x2_factorial.py \
  --dataset all \
  --conditions all \
  --output experiments/results/benchmark_20260801.csv

# Single condition only (either runner)
uv run python experiments/run_2x2_factorial.py \
  --dataset hotpotqa \
  --llm mistral \
  --retrieval dense \
  --output experiments/results/hotpotqa_mistral_dense.csv
```

> ⚠️ **Expected runtime:** The full 800-query experiment takes approximately 4–8 hours on consumer hardware (CPU inference). Run as an overnight job. A `--resume` flag resumes from a checkpoint if the run is interrupted.

After the experiment completes, run the statistical analysis:

```bash
# Docker
docker compose exec ragscope uv run python experiments/analyse_results.py \
  --input experiments/results/benchmark_20260801.csv \
  --output experiments/results/analysis_report.md

# Local
uv run python experiments/analyse_results.py \
  --input experiments/results/benchmark_20260801.csv \
  --output experiments/results/analysis_report.md
```

This produces a Markdown report including descriptive statistics, Cohen's d effect sizes for all pairwise metric comparisons, and Pearson correlation coefficients between latency and RAGAS scores.

---

## Docker Reference

### Service Overview

The `docker-compose.yml` defines two services:

| Service | Image | Port | Role |
|---|---|:---:|---|
| `ragscope` | `./docker/app/Dockerfile` | 8501 | Main application (pipeline + dashboard) |
| `chromadb` | `chromadb/chroma:latest` | 8000 | Persistent vector store |

The `docker-compose.override.yml` (dev mode) adds hot-reload volume mounts to `ragscope` and adds one new service:

| Service | Image | Port | Role |
|---|---|:---:|---|
| `jupyter` *(dev only)* | `./docker/app/Dockerfile`, target `jupyter` (separate image, not `ragscope:latest`) | 8888 | Interactive notebook environment for exploratory analysis. `jupyterlab`/`matplotlib`/`seaborn`/`ipywidgets` are baked into this image at **build** time via a dedicated Dockerfile stage — not installed at container start — so `docker compose up` never re-downloads them. |

### Common Commands

```bash
# Start all services (detached)
docker compose up -d

# View logs for a specific service
docker compose logs -f ragscope

# Open a shell in the app container
docker compose exec ragscope bash

# Run a one-off uv command inside the container
docker compose exec ragscope uv run python -c "from config.settings import settings; print(settings.ollama_base_url)"

# Run the test suite inside Docker
docker compose exec ragscope uv run pytest tests/ -v

# Rebuild the app image after code changes (Compose watch handles this in dev)
docker compose build ragscope

# Hard reset — removes all containers, networks, and volumes
docker compose down -v --remove-orphans
```

### Development Mode (Hot Reload)

For active development, use the override file which mounts the source directory as a volume, so code changes are reflected without rebuilding the image:

```bash
docker compose -f docker-compose.yml -f docker-compose.override.yml up
```

The override also enables Streamlit's `--server.runOnSave` flag, so the dashboard reloads automatically on file save.

### Dockerfile Summary

`docker/app/Dockerfile` (used by both the `ragscope` and dev-only `jupyter` services) is a two-stage build:

```dockerfile
# Stage 1 — builder: uv-managed dependency install (cached unless pyproject.toml / uv.lock change)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
RUN apt-get install -y build-essential gcc g++ zlib1g-dev   # native deps for some packages
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev           # deps only, no notebooks extra
COPY . .
RUN uv sync --frozen --no-dev                                 # then install the project itself

# Stage 2 — runtime: slim Python image, no uv/build tools carried over
FROM python:3.12-slim-bookworm AS app
COPY --from=builder /app /app
COPY docker/scripts/entrypoint.sh /entrypoint.sh
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
ENTRYPOINT ["/entrypoint.sh"]   # waits for chromadb + host Ollama, then starts Streamlit
```

> The `--frozen` flag ensures Docker builds are fully reproducible: the exact versions pinned in `uv.lock` are installed, and the build will fail if the lockfile is out of sync with `pyproject.toml`. The production image deliberately does **not** install the `notebooks` extra (`jupyterlab`/`matplotlib`/`seaborn`/`ipywidgets`) — a separate `builder-notebooks` → `jupyter` stage pair (not shown above) builds those into a distinct image for the dev-only `jupyter` service instead, baked in at build time rather than installed at container start. Both `docker-compose.yml` and `docker-compose.override.yml` pin an explicit `target:` for their respective services, since Docker builds the last stage in a Dockerfile by default when none is given.

---

## Evaluation Metrics

RAGScope computes the following metrics per query using the RAGAS framework (Es et al., 2023):

| Metric | Range | Description |
|---|:---:|---|
| **Context Relevance** | 0–1 | Proportion of retrieved context that is pertinent to the query. Low scores indicate retrieval noise. |
| **Answer Faithfulness** | 0–1 | Degree to which every claim in the generated answer is grounded in the retrieved context. Low scores indicate hallucination. |
| **Answer Correctness** | 0–1 | Semantic alignment of the generated answer with the ground-truth reference answer. |
| **Hallucination Risk Score** | 0–1 | Composite score derived from faithfulness and context relevance. Higher = greater risk. Computed as `1 - (0.6 × faithfulness + 0.4 × context_relevance)`. |

Telemetry metrics collected alongside evaluation:

| Metric | Unit | Description |
|---|---|---|
| **Retrieval Latency** | ms | Time from query submission to final retrieved chunk |
| **Generation Latency** | ms | Time from context injection to final token |
| **Evaluation Latency** | ms | Time spent computing RAGAS metrics (0 if evaluation is disabled) — often the *dominant* term, since it requires one or more auxiliary judge-LLM calls |
| **End-to-End Latency** | ms | Total pipeline wall-clock time — retrieval + generation + evaluation |
| **Prompt Tokens** | count | Tokens in the full prompt (query + retrieved context) |
| **Completion Tokens** | count | Tokens in the generated response |
| **Estimated Cost** | USD | Projected cost based on configurable token rate table |

---

## Experimental Design

The core experiment follows a **2 × 2 fully factorial design**:

|  | **Dense Retrieval** | **Hybrid Retrieval** |
|---|:---:|:---:|
| **Llama 3 (8B)** | Condition A | Condition B |
| **Mistral 7B** | Condition C | Condition D |

- **200 queries** per condition (60 MS MARCO + 40 NQ + 100 HotpotQA)
- **800 total query-result records** with full telemetry and RAGAS annotation
- **Statistical analysis:** Descriptive statistics, Cohen's d effect sizes, Pearson correlations
- **Auxiliary LLM for RAGAS:** Qwen2.5 (7B) — deliberately a third model, distinct from both generator LLMs under evaluation, and held constant across all four conditions

---

## Results

> 🔬 *Results will be populated upon completion of the benchmark evaluation (target: August 2026).*

Preliminary findings and the full analysis report will be available in:
- `experiments/results/analysis_report.md`
- `notebooks/03_hallucination_analysis.ipynb`
- Chapter 4 of the dissertation (submitted September 2026)

---

## Limitations

The following limitations are acknowledged and discussed in full in the dissertation:

- **Domain specificity:** The evaluation corpus is drawn from general-domain benchmarks (Wikipedia, web documents); findings may not generalise to specialist domains (e.g., clinical, legal).
- **RAGAS judge bias:** RAGAS metrics are computed using an LLM judge, which may share biases with the evaluated model - particularly when the same model serves both roles. Mitigation: the judge model (Qwen 2.5) is held constant and separated from the evaluated LLM configurations where possible.
- **Hardware constraints:** Local inference on consumer hardware does not replicate production-scale latency profiles. Reported latency figures should be interpreted comparatively within experimental conditions, not as absolute production benchmarks.
- **Timeline constraints:** The 5.5-month dissertation window limits the scale of experimentation; a larger query benchmark would strengthen statistical conclusions.

---

## Research Outputs

| Output | Status | Location |
|---|---|---|
| MSc Research Proposal | ✅ Complete | Submitted separately — not tracked in this repository |
| Literature Review | 🔄 In progress | Submitted separately — not tracked in this repository |
| System Architecture | ✅ Complete | `docs/architecture.md`, `docs/ragscope_system_architecture.svg` |
| Platform Implementation | ✅ Complete | `pipeline/`, `evaluation/`, `dashboard/` |
| Benchmark Experiment | 🔄 In progress | `experiments/results/` |
| Dissertation | ⏳ Submission Sep 2026 | — |

---

## Citation

If you use RAGScope in your own research, please cite:

```bibtex
@mastersthesis{ragscope2026,
  author  = {Samuel Adole},
  title   = {Real-Time Observability and Evaluation Platform for Retrieval-Augmented Generation Systems},
  school  = {Leeds Beckett University},
  year    = {2026},
  month   = {September},
  type    = {{MSc} Dissertation},
  note    = {School of Built Environment, Engineering and Computing}
}
```

### Key References

- Barnett et al. (2024) — Seven failure points in RAG engineering
- Es et al. (2023) — RAGAS: Automated evaluation of RAG
- Hevner et al. (2004) — Design Science in information systems research
- Lewis et al. (2020) — Retrieval-augmented generation for knowledge-intensive NLP
- Peffers et al. (2007) — Design Science Research Methodology

Full bibliography available in the dissertation.

---

## Licence

This project is licensed under the **MIT Licence**. See [LICENSE](LICENSE) for details.

Datasets used in this research are subject to their own licences:
- MS MARCO: MIT
- Natural Questions: CC BY-SA 3.0
- HotpotQA: CC BY-SA 4.0

---

## Acknowledgements

This project was developed as part of an MSc Data Science dissertation at **Leeds Beckett University**, School of Built Environment, Engineering and Computing.

Thanks to the open-source communities behind [RAGAS](https://github.com/explodinggradients/ragas), [ChromaDB](https://github.com/chroma-core/chroma), [Ollama](https://github.com/ollama/ollama), [Streamlit](https://github.com/streamlit/streamlit), [uv / Astral](https://github.com/astral-sh/uv), and [Docker](https://github.com/docker), whose tools make research like this tractable.

---

<div align="center">

*Built for MSc Data Science · Leeds Beckett University · 2026*

</div>