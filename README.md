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
  - [Building the Knowledge Graph (Graph Retrieval)](#building-the-knowledge-graph-graph-retrieval)
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

- **RQ1 — Metric Calibration:** How reliable is a real-time composite risk score in estimating actual answer correctness within RAG systems?
- **RQ2 — Fault Localisation:** How effectively can real-time metrics isolate the cause of a RAG failure to either the retrieval or generation stage?
- **RQ3 — Configuration Trade-offs:** How do dense and hybrid retrieval strategies, when paired with Llama 3 (8B) and Mistral (7B), trade off between output quality (faithfulness and hallucination risk) and system efficiency (latency and token use)?

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

The platform is organised into six stages, each mapped to a specific research question or SMART objective — see `docs/images/ragscope_system_architecture.svg` for the full diagram (and `docs/architecture.md` for the detailed prose description of every module).

```
 1. Data          2. RAG Query Pipeline        3. Evaluation
 ────────    ▶    ─────────────────────   ▶    ─────────────
 BioASQ corpus    Dense/Hybrid retrieval        RAGAS Runner (judge: qwen2.5:7b)
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
- Read-only ChromaDB knowledge-base browser with corpus composition breakdown, a 3D PCA projection of sampled embeddings, and a semantic search preview
- Export of session data to CSV for downstream analysis

### 🗂️ Benchmark Pipeline
- Pre-built BioASQ loaders covering three evaluation roles (Phase A retrieval, factoid, summary)
- Stratified query sampling utilities
- Ground-truth answer alignment for RAGAS answer correctness computation

---

## Datasets

A single benchmark dataset, **BioASQ**, is used for evaluation — accessed via `rag-datasets/rag-mini-bioasq`, a freely downloadable HuggingFace mirror derived from the official BioASQ Task 11b training release (CC BY 2.5). The official release requires registration at bioasq.org and includes a ~23M-abstract PubMed corpus; this mirror provides a bounded, dissertation-scale derivative (40,221 passages, 4,719 QA pairs) instead. See [`docs/datasets.md`](docs/datasets.md) for the full methodology, including how the three evaluation roles are reconstructed from BioASQ's answer shapes (this mirror carries no native question-type label).

### BioASQ
- **Source:** [huggingface.co/datasets/rag-datasets/rag-mini-bioasq](https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq)
- **Licence:** CC BY 2.5
- **Usage in this project:** 27,972 PubMed-abstract passages (cleaned from a raw 40,221) indexed as the retrieval corpus; 200 queries split across three roles:
  - **Phase A** (60 queries, 30 single-relevant + 30 multi-relevant) — passage retrieval over the PubMed corpus
  - **Factoid** (40 queries, short exact answers) — precise entity/phrase-level question answering
  - **Summary** (100 queries, 50 yes/no + 50 long-form) — multi-passage synthesis
- **Why:** A single, domain-realistic corpus — biomedical questions authored by domain experts against PubMed abstracts — spanning three complementary evaluation conditions (retrieval / factual QA / synthesis), matching BioASQ's own Phase A / Phase B task structure

| Role | Query Type | Corpus Source | Queries Used |
|---|---|---|:---:|
| Phase A | Passage retrieval | PubMed abstracts | 60 |
| Factoid | Short exact-answer QA | PubMed abstracts | 40 |
| Summary | Multi-passage synthesis | PubMed abstracts | 100 |
| **Total** | | | **200** |

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
│   │   └── bioasq.py              # BioASQ corpus + 3-role query loader
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
│   │   └── 05_knowledge_base.py   # Read-only ChromaDB browser + 3D embedding view + semantic search preview
│   └── components/
│       ├── theme.py               # Shared icons (Material Symbols / inline SVG) + CSS theme
│       ├── metric_cards.py        # Metric display components
│       ├── latency_chart.py       # Latency visualisation
│       ├── heatmap.py             # RAGAS score heatmaps
│       └── vector_space.py        # Pure-numpy PCA + 3D embedding scatter
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

**5. Ingest the benchmark dataset**

Run this once to download and embed the BioASQ passages into ChromaDB:

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
  "Which protein does empagliflozin inhibit?" \
  --llm llama3 \
  --retrieval hybrid \
  --top-k 5

# Local
uv run python pipeline/rag.py \
  "Which protein does empagliflozin inhibit?" \
  --llm llama3 \
  --retrieval hybrid \
  --top-k 5
```

> The query is a positional argument, not a `--query` flag — `pipeline/rag.py` is a Typer CLI with `query` declared as `typer.Argument`.

**Sample output:**

```
────────────────────────────────────────────────────────
 Query : Which protein does empagliflozin inhibit?
────────────────────────────────────────────────────────
 Retrieved chunks : 5 (strategy: hybrid)
   [1] score=0.016 | empagliflozin reduces HG induced inflammatory and fibrotic markers by blocking…
   [2] score=0.016 | 0.80 - 1.01). Empagliflozin was well tolerated under fed and fasting condition…
   [3] score=0.016 | OBJECTIVES: Empagliflozin is an orally available, potent and highly selective i…

 Answer : According to Passage 3, empagliflozin is an orally available, potent and highly selective inhibitor of the sodium glucose cotransporter 2 (SGLT2).

 ── Telemetry ──────────────────────────────────────
  Retrieval latency  : 328.9 ms
  Generation latency : 6857.2 ms
  Evaluation latency : 20534.3 ms
  End-to-end latency : 27725.0 ms
  Prompt tokens      : 1520
  Completion tokens  : 37
  Estimated cost     : $0.000000

 ── RAGAS Evaluation ───────────────────────────────
  Context Relevance  : 0.7500
  Answer Faithfulness: 1.0000
  Answer Correctness : N/A (no ground truth)
  Hallucination Risk : LOW  (0.1000)
────────────────────────────────────────────────────────
 Telemetry → telemetry/store/ee89ca0c-83de-489c-ab93-e4f865dae4ad.json
────────────────────────────────────────────────────────
```

> Hybrid scores are Reciprocal Rank Fusion values (around 0.016), not cosine similarities. Answer correctness needs a reference answer, so it shows `N/A` for an ad-hoc question.

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
- **Knowledge Base** — read-only browser of the ChromaDB collection: corpus composition by dataset, a 3D PCA projection of sampled embeddings coloured by dataset, a paginated chunk browser, and a semantic search preview

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
  --dataset bioasq_summary \
  --llm mistral \
  --retrieval dense \
  --output experiments/results/bioasq_summary_mistral_dense.csv
```

> ⚠️ **Expected runtime:** The full 800-query experiment takes on the order of half a day on consumer hardware (CPU inference) — RAGAS judge latency dominates per-query time. Run as a background/overnight job. A `--resume` flag resumes from a checkpoint if the run is interrupted.

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

### Building the Knowledge Graph (Graph Retrieval)

The `graph` retrieval strategy reads a knowledge graph that is built offline. Build it once before selecting `graph`. Dense and hybrid retrieval do not need it.

The builder uses [docling-graph](https://github.com/docling-project/docling-graph) 1.9.1 with a local Ollama model to extract biomedical entities (diseases, genes/proteins, chemicals/drugs) and their relations from each chunk. docling-graph is a normal project dependency, so `uv sync` installs it and the builder runs in the same environment as everything else. The commands below run from the repository root on the host, like the *Local* variants above.

**Before you start**

- `data/bm25_corpus.jsonl` must exist. The ingestion step writes it. On the Docker path it is written inside the container, so copy it out first:

  ```bash
  docker compose cp ragscope:/app/data/bm25_corpus.jsonl data/bm25_corpus.jsonl
  ```

- Ollama must be running on the host, with the extractor model pulled:

  ```bash
  ollama pull llama3.2           # ~2 GB
  ```

The builder uses the Ollama address from `.env` (`OLLAMA_BASE_URL`), like the rest of the pipeline.

**1. Sample the chunks to extract**

```bash
uv run python graph/spikes/make_sample.py 300
```

This writes one Markdown file per chunk into `data/graph/sample/`, replacing any previous sample. Chunks are picked at an even stride across the whole corpus, with no reference to the benchmark queries, so the graph is not biased towards the questions it will be tested on. The shipped graph uses 300 chunks, and this command reproduces exactly that sample.

**2. Run the extraction**

```bash
uv run python graph/spikes/build_subset.py 8     # argument: concurrent requests (default 8)
```

Progress is printed every 25 chunks. When extraction finishes, the builder writes its output to `data/graph/`, which is where the retriever reads from, so there is nothing to copy:

| File | Contents |
|---|---|
| `graph.json` | The graph, as NetworkX node-link JSON |
| `node_to_chunks.json` | For each node, the chunk IDs it was extracted from. This is how retrieval maps graph hits back to passages. |
| `manifest.json` | Build statistics: chunks attempted and succeeded, failures, timing, node and edge counts |

A finished build **replaces the current graph**. Nothing is written until extraction completes, so an interrupted build leaves the existing graph untouched. To build somewhere else, set `GRAPH_PATH` and `GRAPH_NODE_CHUNKS_PATH` in `.env` (or the shell). The builder and the retriever both read those settings, and the sample folder sits next to `GRAPH_PATH`.

> ⏱️ **Expected runtime:** about 1 hour for 300 chunks (11.8 s per chunk on average on an Apple M5 with `llama3.2` on the GPU). Individual chunks vary widely; some take over a minute. Above about 2 concurrent requests, raising the concurrency makes little difference, because a single request already keeps the GPU busy. A chunk that fails extraction is skipped and listed under `failures` in the manifest (2 of 300 in the shipped build). Extraction runs at temperature 0 with a fixed seed, so the same sample gives the same graph. Building over the full corpus is not practical on a laptop; it projects to roughly 100 hours.

**3. Use it**

```bash
uv run python pipeline/rag.py "Which protein does empagliflozin inhibit?" --retrieval graph
```

In the dashboard, choose **graph** under *Retrieval Strategy* in the Live Monitor sidebar. The retriever loads the graph on every run, but restart the dashboard so the Knowledge Base page picks up the new graph.

- **Local:** nothing else to do.
- **Docker (dev override):** the source directory is mounted, so the container sees `data/graph/` straight away.
- **Docker (production compose):** the graph is copied into the image at build time, so rebuild: `docker compose build ragscope && docker compose up -d`.

> By default, graph results are fused with dense retrieval (`GRAPH_FUSE_DENSE=true`). While the graph covers only a small part of the corpus, fusion can reorder what dense retrieval found but can't bring in a passage that only the graph found. Set `GRAPH_FUSE_DENSE=false` to see graph-only results.

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

- **200 queries** per condition (60 BioASQ Phase A + 40 factoid + 100 summary)
- **800 total query-result records** with full telemetry and RAGAS annotation
- **Statistical analysis:** Descriptive statistics, Cohen's d effect sizes, Pearson correlations
- **Auxiliary LLM for RAGAS:** Qwen2.5 (7B) — deliberately a third model, distinct from both generator LLMs under evaluation, and held constant across all four conditions

---

## Results

The 2 × 2 benchmark produced 800 query-result records (200 queries × 4 conditions). 797 of them (99.6%) received complete RAGAS scores; the other three failed evaluation and were kept for the latency analysis only.

| Research question | Finding |
|---|---|
| **RQ1 — Metric calibration** | The hallucination risk score works as a coarse safe/unsafe filter but does not estimate correctness finely. LOW-risk answers scored a mean correctness of 0.516, nearly double the MEDIUM (0.270) and HIGH (0.290) bands, but the score did not separate MEDIUM from HIGH. |
| **RQ2 — Fault localisation** | Stage-level telemetry exposed the main bottleneck: RAGAS evaluation took **85.7%** of end-to-end latency, against 14.1% for generation and 0.2% for retrieval. It also showed that Llama 3 generated faster than Mistral, although its end-to-end latency was slightly higher. |
| **RQ3 — Configuration trade-offs** | Hybrid retrieval gave no meaningful gain over dense: every pooled effect was negligible (context relevance *d* = +0.036, faithfulness *d* = −0.063). The LLM mattered more: Llama 3 was more faithful than Mistral (*d* = +0.190) with lower hallucination risk (*d* = −0.131), with no advantage in answer correctness (*d* = −0.046). No configuration won on every dimension. |

Cohen's *d* thresholds: |*d*| < 0.20 negligible, 0.20–0.49 small, 0.50–0.79 medium, ≥ 0.80 large.

**Graph retrieval (proof of concept).** Graph retrieval was added after the benchmark, as a platform extension, and is not one of the four experimental conditions; no quality comparison has been run. Over a 298-chunk graph, it found starting entities for 74 of the 200 benchmark queries, and graph-only retrieval took a median of 19.9 ms. See [Building the Knowledge Graph](#building-the-knowledge-graph-graph-retrieval).

Full results and analysis:

- Chapter 4 of the dissertation (submitted September 2026)
- `notebooks/03_hallucination_analysis.ipynb`
- `experiments/results/`: the benchmark CSV, the analysis report and the figures. This folder is git-ignored; [Running the Benchmark Evaluation](#running-the-benchmark-evaluation) regenerates it.

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
| Literature Review | ✅ Complete | Submitted separately — not tracked in this repository |
| System Architecture | ✅ Complete | `docs/architecture.md`, `docs/images/ragscope_system_architecture.svg` |
| Platform Implementation | ✅ Complete | `pipeline/`, `evaluation/`, `dashboard/` |
| Benchmark Experiment | ✅ Complete | `experiments/results/` |
| Dissertation | ✅ Complete | — |
| Publication | 🔄 In progress | — |

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

The dataset used in this research is subject to its own licence:
- BioASQ (via `rag-datasets/rag-mini-bioasq`): CC BY 2.5

---

## Acknowledgements

This project was developed as part of an MSc Data Science dissertation at **Leeds Beckett University**, School of Built Environment, Engineering and Computing.

Thanks to the open-source communities behind [RAGAS](https://github.com/explodinggradients/ragas), [ChromaDB](https://github.com/chroma-core/chroma), [Ollama](https://github.com/ollama/ollama), [Streamlit](https://github.com/streamlit/streamlit), [docling-graph](https://github.com/docling-project/docling-graph), [uv / Astral](https://github.com/astral-sh/uv), and [Docker](https://github.com/docker), whose tools make research like this tractable.

---

<div align="center">

*Built for MSc Data Science · Leeds Beckett University · 2026*

</div>