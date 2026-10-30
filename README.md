<div align="center">

# 🔭 RAGScope
### Real-Time Observability & Evaluation Platform for Retrieval-Augmented Generation Systems

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![uv](https://img.shields.io/badge/uv-package_manager-DE5FE9?style=flat-square&logo=astral&logoColor=white)](https://docs.astral.sh/uv)
[![Docker](https://img.shields.io/badge/Docker-compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.55-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-orange?style=flat-square)](https://trychroma.com)
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
  - [Quick Start (CPU-only Laptop)](#quick-start-cpu-only-laptop)
  - [Prerequisites](#prerequisites)
  - [Option A - Docker](#option-a--docker)
  - [Option B - Local with uv](#option-b--local-with-uv)
  - [Configuration](#configuration)
- [Usage](#usage)
  - [Running the RAG Pipeline](#running-the-rag-pipeline)
  - [Launching the Dashboard](#launching-the-dashboard)
  - [Running the Benchmark Evaluation](#running-the-benchmark-evaluation)
  - [Asking Questions About Your Own Documents](#asking-questions-about-your-own-documents)
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

- **RQ1 — Metric Calibration:** To what extent does a real-time composite hallucination risk score predict actual answer correctness within RAG systems?
- **RQ2 — Fault Localisation:** How effectively can real-time observability metrics isolate the root cause of a RAG failure to either the retrieval or generation stage?
- **RQ3 — Configuration Trade-offs:** How do dense and hybrid retrieval strategies, when paired with different open-weight LLMs (Llama 3 8B and Mistral 7B), balance output quality against system efficiency?

---

## Research Context

This platform was developed as the primary artefact for an MSc Data Science dissertation at Leeds Beckett University. The research follows the **Design Science Research Methodology** (Peffers et al., 2007), in which the construction and rigorous empirical evaluation of a novel software artefact constitutes a legitimate and substantive contribution to knowledge.

### The Problem

Despite the maturity of evaluation frameworks like RAGAS (Es et al., 2024), a critical gap remains:

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
 BioASQ corpus    Dense/hybrid/graph retrieval  RAGAS Runner (judge: qwen2.5:7b)
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
 (RQ3 / Objective 4)                                   ▼

                                                  5. Dashboard
                                                  ─────────────
                                                  Live Monitor · Query
                                                  Explorer · Benchmark
                                                  Results · Knowledge Base
                                                  (Objective 2)
```

---

## Features

### 🔬 Real-Time Evaluation
- Per-query RAGAS metric computation (context relevance, answer faithfulness, answer correctness)
- Composite hallucination risk scoring derived from faithfulness and relevance signals
- Evaluation runs alongside query execution — no separate batch job required

### 📡 Telemetry Collection
- End-to-end latency profiling with stage-level decomposition (retrieval, generation and evaluation)
- Token usage logging (prompt tokens, completion tokens, total)
- Per-query cost estimation based on configurable per-token rate tables
- Structured JSON telemetry store for reproducibility and offline analysis

### 🔁 Comparative Experimentation
- Pluggable retrieval strategies: **dense retrieval** (ChromaDB vector search), **hybrid retrieval** (BM25 + dense via Reciprocal Rank Fusion) and **graph retrieval** (knowledge-graph traversal, added after the experiment)
- Any Ollama model as the generator, set with `GENERATOR_MODELS`; the experiment used **Llama 3 (8B)** and **Mistral 7B**. Everything runs locally, with no API key
- 2 × 2 factorial experiment runner with automated result logging

### 📊 Interactive Dashboard
- Live metric time series for active query sessions, with results persisted across page interaction
- Benchmark Results page comparing the four conditions: summary statistics, effect sizes, risk profile, latency and token breakdown, and metric correlations
- Retrieved document chunk viewer with colour-graded per-chunk similarity scores
- Sankey diagram of the full pipeline process flow — every instrumented stage (embedding, retrieval sub-stages, generation, RAGAS evaluation) sized by actual wall-clock duration for a single query
- The generated answer appears as soon as it exists, while RAGAS evaluation is still running
- Knowledge-base page: upload your own PDF and text files, and browse the BioASQ collection with its corpus composition, a 3D PCA projection of sampled embeddings, a semantic search preview and the knowledge graph
- CSV export of benchmark results for downstream analysis

### 🗂️ Benchmark Pipeline
- Pre-built BioASQ loaders covering three evaluation roles (Phase A retrieval, factoid, summary)
- Stratified query sampling utilities
- Ground-truth answer alignment for RAGAS answer correctness computation

### 📄 Your Own Documents
- Ingest PDF, plain-text and Markdown files into a knowledge base kept separate from the benchmark
- PDFs are read page by page, so every retrieved chunk cites its file and page

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
| LLM Inference | [Ollama](https://ollama.com) | Local model serving: Llama 3 8B and Mistral 7B in the research configuration, 3B models in the CPU default. Runs on the **host** machine, not as a Compose service (containers reach it via `host.docker.internal`) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Dense vector generation |
| Vector DB | [ChromaDB](https://trychroma.com) | Embedding storage & similarity search: embedded in the app by default, or a separate server (the Compose service) |
| Sparse Retrieval | `rank_bm25` | BM25 scoring for hybrid retrieval |
| Hybrid Fusion | Custom RRF implementation | Reciprocal Rank Fusion |
| Evaluation | [RAGAS](https://docs.ragas.io) | Context relevance, faithfulness, correctness (judge: Qwen2.5 7B in the research configuration, `qwen2.5:3b` by default) |
| Telemetry | Custom JSON logger | Query-level instrumentation store |
| Dashboard | [Streamlit](https://streamlit.io) | Interactive real-time visualisation |
| Data Processing | `pandas`, `numpy` | Benchmark loading and analysis |
| Visualisation | `plotly` | Dashboard charts, heatmaps, and Sankey process-flow diagrams |
| Statistics | `statsmodels` | OLS trendlines on dashboard scatter plots |
| Knowledge graph | `networkx`; [docling-graph](https://github.com/docling-project/docling-graph) (optional, build only) | Graph retrieval; offline entity and relation extraction |
| Documents | `pypdf` | Text extraction from your own PDFs |

---

## Project Structure

```
ragscope/
│
├── README.md
├── CITATION.cff                   # Citation metadata ("Cite this repository" on GitHub)
├── pyproject.toml                 # Project metadata & dependencies (uv); optional extra: graph
├── uv.lock                        # Pinned lockfile — commit this for reproducibility
├── .env.example                   # Default configuration: small models, CPU-only, embedded ChromaDB
├── .env.research.example          # The dissertation configuration (Llama 3, Mistral, Qwen2.5 7B judge)
├── .dockerignore
├── .gitignore
├── LICENSE
│
├── docker-compose.yml             # The two services, ragscope (app) and chromadb, plus watch mode for development (Ollama runs on the host)
│
├── docker/                        # Per-service Dockerfiles
│   ├── app/
│   │   └── Dockerfile             # RAGScope app image (Python 3.12, uv-installed deps)
│   └── scripts/
│       └── entrypoint.sh          # Container startup: waits for host Ollama + ChromaDB to be reachable
│
├── config/
│   └── settings.py                # All settings, read from .env
│
├── scripts/
│   ├── fetch_index.py             # Download the prebuilt index instead of running ingestion
│   ├── package_index.py           # Maintainer tool: build the index archive for a release
│   └── fetch_example_documents.py # Five CC BY papers from PubMed Central, to try document ingestion
│
├── data/                          # Dataset loading and preprocessing
│   ├── loaders/
│   │   ├── bioasq.py              # BioASQ corpus + 3-role query loader
│   │   └── documents.py           # Your own PDF / .txt / .md files
│   ├── preprocessing/
│   │   ├── chunker.py             # Text chunking strategies
│   │   └── cleaner.py             # Text normalisation utilities
│   ├── documents/                 # Your own files for --corpus documents (git-ignored)
│   ├── graph/                     # Knowledge graph used by graph retrieval (git-ignored)
│   └── raw/                       # Downloaded datasets (git-ignored)
│
├── pipeline/                      # Core RAG pipeline
│   ├── ingestion.py               # Document ingestion orchestrator
│   ├── knowledge_base.py          # BioASQ benchmark vs. your documents: collection + BM25 file
│   ├── embeddings.py              # Embedding generation wrapper
│   ├── vectorstore.py             # ChromaDB interface (embedded or server)
│   ├── retrieval/
│   │   ├── dense.py               # Dense retrieval (vector similarity)
│   │   ├── hybrid.py              # Hybrid retrieval (BM25 + RRF)
│   │   └── graph.py               # Graph retrieval (entity seeding + Personalized PageRank)
│   ├── generation/
│   │   ├── __init__.py            # build_llm_client(): picks the client for a model tag
│   │   ├── llm_client.py          # Ollama API wrapper and the generic RAG prompt
│   │   ├── llama3.py              # Llama 3 (8B) prompt template
│   │   └── mistral.py             # Mistral 7B prompt template
│   └── rag.py                     # End-to-end query pipeline orchestrator
│
├── graph/                         # Offline knowledge-graph builder (uv sync --extra graph)
│   ├── sample_chunks.py           # Pick the chunks to extract
│   ├── build_graph.py             # Extract entities and relations with docling-graph
│   ├── builder_config.py          # Extraction settings and paths
│   ├── biomed_template.py         # Biomedical entity and relation schema
│   ├── lightrag_compare.py        # LightRAG baseline comparison
│   └── lightrag_results.md        # Its results
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
│   └── benchmark.py               # 200-query benchmark runner
│
├── dashboard/                     # Streamlit observability dashboard
│   ├── app.py                     # Main Streamlit application entry point
│   ├── pages/
│   │   ├── 01_live_monitor.py     # Ask a question; answer, scores, pipeline flow Sankey, graph path
│   │   ├── 02_query_explorer.py   # Per-query drill-down view + pipeline flow Sankey
│   │   ├── 03_benchmark_results.py # Benchmark results: the four conditions compared
│   │   └── 04_knowledge_base.py   # Upload your documents; browse the BioASQ collection and graph
│   └── components/
│       ├── theme.py               # Shared icons (Material Symbols / inline SVG) + CSS theme
│       ├── metric_cards.py        # Metric display components
│       ├── latency_chart.py       # Latency visualisation
│       ├── graph_view.py          # Interactive retrieval-path graph
│       └── vector_space.py        # Pure-numpy PCA + 3D embedding scatter
│
├── experiments/                   # Experimental scripts
│   ├── run_2x2_factorial.py       # Full 2×2 factorial experiment runner
│   ├── analyse_results.py         # Statistical analysis (Cohen's d, etc.)
│   └── results/                   # Experiment output CSVs (git-ignored)
│
├── tests/                         # Unit and integration tests (uv run pytest)
│
└── docs/                          # Extended documentation
    ├── architecture.md            # Full module-by-module architecture reference
    ├── datasets.md                # Dataset selection, sampling, and provenance
    ├── evaluation_metrics.md      # RAGAS + hallucination risk metric definitions
    └── images/
        └── ragscope_system_architecture.svg  # System architecture diagram
```

---

## Getting Started

### Quick Start (CPU-only Laptop)

The shortest path to a working dashboard. It needs no GPU, no Docker and no database server: small (3B) models, ChromaDB embedded in the app, and a prebuilt index instead of the ingestion step.

```bash
# 1. Tools: uv (https://docs.astral.sh/uv) and Ollama (https://ollama.com)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Code and Python environment
git clone https://github.com/samueladole/MDATS79.git ragscope
cd ragscope
uv sync                                    # CPU-only PyTorch on Linux; no CUDA download

# 3. Models (about 4 GB in total)
ollama pull llama3.2:3b                    # generator
ollama pull qwen2.5:3b                     # RAGAS judge

# 4. Configuration and data
cp .env.example .env                       # CPU defaults
uv run python scripts/fetch_index.py       # prebuilt index (~350 MB), checked against its SHA-256
                                           # (or build it: uv run python pipeline/ingestion.py --corpus all, 20–40 min)

# 5. Dashboard
uv run streamlit run dashboard/app.py      # http://localhost:8501
```

Open **Live Monitor**, ask a biomedical question such as *Which protein does empagliflozin inhibit?*, and press **Run Query**.

> ⏱️ **What to expect on CPU only.** Measured over 8 benchmark queries with Ollama in a CPU-only container (10 cores of an Apple M5, 8 GB memory):
>
> - The answer appears after about **25 s** (median 23 s, range 14–41 s).
> - The RAGAS scores follow about **100 s** later (median 102 s, range 75–161 s), because the judge makes three model calls per query.
> - End to end, a query takes about **2 minutes** (median 121 s).
>
> Switch off **Run RAGAS Evaluation** in the sidebar for quick exploration. Older or smaller laptops will be slower. With 8 GB of memory, keeping the generator and judge loaded together is tight: the model server crashed twice in nine queries. Close other memory-heavy applications, or use `GENERATOR_MODELS=qwen2.5:3b` and `DEFAULT_LLM=qwen2.5:3b` so the generator and judge are the same model. The judge then scores its own answers, so treat the scores with more caution.

To reproduce the dissertation configuration (Llama 3 8B and Mistral 7B, Qwen2.5 7B judge) use `cp .env.research.example .env` instead, and pull `llama3`, `mistral` and `qwen2.5:7b`. Those models need about 14 GB of disk and are practical only with a GPU or Apple silicon. The published results were produced on an Apple M5 MacBook Pro (16 GB) with Ollama running the models on its integrated GPU.

### Prerequisites

| Requirement | Version | Install |
|---|---|---|
| [Docker](https://docs.docker.com/get-docker/) | ≥ 26.0 | Optional: only for the Docker path (Option A) |
| [Docker Compose](https://docs.docker.com/compose/) | ≥ 2.27 | Optional: bundled with Docker Desktop |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | ≥ 0.4 | Required for the quick start and the local path (Option B) |
| [Ollama](https://ollama.com) | Latest | **Required for every path** — Ollama always runs on the host, never inside a container. Install it and pull the models first (see below). |
| RAM | ≥ 8 GB (CPU defaults) | 16 GB for the research configuration; 32 GB recommended for running both 7–8B models |
| Disk | ≥ 6 GB free (CPU defaults) | Research configuration: ≥ 25 GB (models ~14 GB + corpus + embeddings + images) |

> **macOS / Linux — install uv in one line:**
> ```bash
> curl -LsSf https://astral.sh/uv/install.sh | sh
> ```
> **Windows:**
> ```powershell
> powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
> ```

---

### Option A — Docker

Docker Compose orchestrates two services automatically: the RAGScope app and ChromaDB (vector store). No manual setup of Python or ChromaDB is required — but **Ollama is not containerised**: it always runs on the host machine (reached via `host.docker.internal`), so it must be installed and have its models pulled *before* starting the stack, for both this path and Option B.

**1. Clone the repository**

```bash
git clone https://github.com/samueladole/MDATS79.git ragscope
cd ragscope
```

**2. Install Ollama and pull the required models (on the host)**

```bash
# Install from https://ollama.com, then:
ollama serve &                # Start the Ollama daemon in the background, if not already running

# CPU defaults (.env.example)
ollama pull llama3.2:3b        # ~2.0 GB
ollama pull qwen2.5:3b         # ~1.9 GB (RAGAS judge model)

# Research configuration (.env.research.example)
ollama pull llama3             # ~4.7 GB
ollama pull mistral            # ~4.4 GB
ollama pull qwen2.5:7b         # ~4.7 GB (RAGAS judge model)
```

The container entrypoint polls this host Ollama instance on startup and fails fast with a diagnostic message if no model is available — it does **not** pull models itself.

**3. Configure environment variables**

```bash
cp .env.example .env              # CPU defaults, or: cp .env.research.example .env
# Edit .env if you need to change defaults (see Configuration below)
```

Inside Docker the app always talks to the `chromadb` service, whatever `CHROMA_MODE` says in `.env`.

**4. Build and start the services**

```bash
docker compose up --build        # add --watch to develop inside Docker (see Development Mode)
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
docker compose exec ragscope python pipeline/ingestion.py \
  --corpus all \
  --chunk-size 512 \
  --chunk-overlap 64
```

> This embeds the 30,850 BioASQ chunks using `all-MiniLM-L6-v2`. Expect 20–40 minutes on CPU. Progress is displayed in the terminal. The vectors go to the `chroma_data` volume, and the BM25 corpus to `data/` on your machine, which Compose mounts into the container.

**6. Stop all services**

```bash
docker compose down          # Stops containers, preserves volumes
docker compose down -v       # Stops and removes all volumes (full reset)
```

---

### Option B — Local with uv

Use this path for active development, or to run without Docker.

**1. Clone and enter the repository**

```bash
git clone https://github.com/samueladole/MDATS79.git ragscope
cd ragscope
```

**2. Install dependencies with uv**

uv reads `pyproject.toml` and creates an isolated virtual environment automatically. There is no separate `venv` activation step required.

```bash
uv sync                      # Installs all dependencies from uv.lock
uv sync --extra graph        # Also installs docling-graph, needed only to BUILD the knowledge graph
```

On Linux, PyTorch comes from PyTorch's CPU-only package index, so no CUDA libraries are downloaded. macOS and Windows use the standard wheels.

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

ollama pull llama3.2:3b      # CPU defaults: generator
ollama pull qwen2.5:3b       # CPU defaults: RAGAS judge
# Research configuration instead: llama3, mistral, qwen2.5:7b
```

**4. Configure**

```bash
cp .env.example .env         # embedded ChromaDB: no database server to start
```

With `CHROMA_MODE=http` (as in `.env.research.example`), start a ChromaDB server first:

```bash
uv run chroma run --path ./data/chroma_store --port 8000
```

**5. Get the index: download it, or build it from source**

```bash
uv run python scripts/fetch_index.py                       # prebuilt, ~350 MB
# or
uv run python pipeline/ingestion.py --corpus all --chunk-size 512 --chunk-overlap 64   # 20–40 min on CPU
```

**6. Run tests to verify setup**

```bash
uv run pytest tests/ -v
```

---

### Configuration

All configuration is read from a single `.env` file, by both Docker Compose and the local uv runner. Two starting points ship with the repository:

| File | For | Generators | RAGAS judge | ChromaDB |
|---|---|---|---|---|
| `.env.example` | Any laptop, CPU only (the default) | `llama3.2:3b`, `qwen2.5:3b` | `qwen2.5:3b` | embedded |
| `.env.research.example` | Reproducing the dissertation | `llama3`, `mistral` | `qwen2.5:7b` | server |

The settings most often changed:

| Setting | Meaning |
|---|---|
| `GENERATOR_MODELS` | Comma-separated Ollama model tags offered in the dashboard and CLI. `llama3` and `mistral` use their own chat templates, as in the experiments; any other tag uses a generic prompt. |
| `DEFAULT_LLM` | The generator selected by default. Must be one of `GENERATOR_MODELS`. |
| `RAGAS_JUDGE_MODEL` | The model that scores each answer. |
| `CHROMA_MODE` | `embedded` keeps the index in `CHROMA_PATH` inside the app; `http` connects to a server at `CHROMA_HOST:CHROMA_PORT`. Docker always uses `http`. |
| `RAGAS_TIMEOUT_SECONDS` | Time allowed per judge call, including time queued behind the other metrics. 180 by default; 600 in the CPU profile. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` locally; Docker sets `host.docker.internal` itself. |
| `RETRIEVAL_STRATEGY`, `TOP_K` | Default retrieval strategy (`dense`, `hybrid`, `graph`) and passages per query. |

See the comments in either file for the rest.

> **Docker vs. local hostnames:** `CHROMA_HOST` differs by path — the real Compose service name (`chromadb`) inside Docker, `localhost` when running locally. `OLLAMA_BASE_URL` differs the other way: `host.docker.internal` inside Docker (Ollama is never a Compose service — it's always on the host), `localhost` when running locally, since there's no container boundary to cross. Compose sets both for the containers.

---

## Usage

Commands below are shown in two variants: **Docker** and **local uv**. They are functionally identical. Inside the container, run `python` directly: the app image carries the project's environment but not `uv`.

### Running the RAG Pipeline

Run a single query through the full pipeline (retrieval + generation + telemetry + evaluation):

```bash
# Docker
docker compose exec ragscope python pipeline/rag.py \
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

> The query is a positional argument, not a `--query` flag — `pipeline/rag.py` is a Typer CLI with `query` declared as `typer.Argument`. `--llm` takes any Ollama model tag (for example `llama3.2:3b`) and defaults to `DEFAULT_LLM`. The sample below is from the research configuration.

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

The dashboard provides four views at `http://localhost:8501`:

- **Live Monitor** — submit a query interactively and watch the full pipeline execute: generated answer, RAGAS scores, colour-graded chunk similarity, and a Sankey diagram of every instrumented stage's wall-clock duration. The answer is shown as soon as it is generated, while the RAGAS scores are still being computed. Choose the **Knowledge base** (BioASQ benchmark or your own documents) in the sidebar
- **Query Explorer** — click any logged query to inspect retrieved chunks, similarity scores, faithfulness breakdowns, latency decomposition, and the same per-query pipeline flow Sankey
- **Benchmark Results** — the four experimental conditions compared over the 200-query benchmark: summary statistics, hallucination risk profile, Cohen's *d* effect sizes, latency and token breakdown, metric correlations, and performance by query role
- **Knowledge Base** — upload your own PDF and text files (see below), and browse the BioASQ collection: corpus composition by dataset, a 3D PCA projection of sampled embeddings coloured by dataset, a paginated chunk browser, a semantic search preview, and the knowledge graph

### Running the Benchmark Evaluation

The full 2 × 2 factorial benchmark (200 queries × 4 conditions = 800 records):

```bash
# Docker
docker compose exec ragscope python experiments/run_2x2_factorial.py \
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

> ⚠️ **Expected runtime:** The full 800-query experiment averaged about 57 s per query, around 13 hours in total, on an Apple M5 MacBook Pro with Ollama on its integrated GPU. On a CPU alone it takes several times longer. RAGAS judge latency dominates per-query time. Run it as a background or overnight job; `--resume` continues from a checkpoint if the run is interrupted.

After the experiment completes, run the statistical analysis:

```bash
# Docker
docker compose exec ragscope python experiments/analyse_results.py \
  --input experiments/results/benchmark_20260801.csv \
  --output experiments/results/analysis_report.md

# Local
uv run python experiments/analyse_results.py \
  --input experiments/results/benchmark_20260801.csv \
  --output experiments/results/analysis_report.md
```

This produces a Markdown report including descriptive statistics, Cohen's d effect sizes for all pairwise metric comparisons, and Pearson correlation coefficients between latency and RAGAS scores.

### Asking Questions About Your Own Documents

RAGScope can ingest your own PDF, plain-text and Markdown files and answer questions about them, with the same retrieval, generation, telemetry and RAGAS scoring as the benchmark.

They go into a separate ChromaDB collection (`ragscope_documents`) and a separate BM25 file, so ingesting them never changes the BioASQ index or the published results.

**In the dashboard:** open **Knowledge Base → My documents**, add files, and press **Save and index**. Then in **Live Monitor**, choose **Knowledge base → My documents**.

**From the command line:**

```bash
# Optional: five open-access (CC BY) papers on SGLT2 inhibitors from PubMed Central
uv run python scripts/fetch_example_documents.py

# Index everything in data/documents/ (or pass --source path/to/folder)
uv run python pipeline/ingestion.py --corpus documents
```

- PDFs are split into one passage per page, and each retrieved chunk shows its file and page.
- Only the PDF's text layer is read, so scanned PDFs are skipped with a warning.
- The collection is rebuilt from the folder on every run: to remove a file, delete it and ingest again.
- Answer correctness needs a reference answer, so it shows N/A for your own questions.
- Graph retrieval is available for the BioASQ benchmark only, because the knowledge graph is built over that corpus.

### Building the Knowledge Graph (Graph Retrieval)

The `graph` retrieval strategy reads a knowledge graph that is built offline. Build it once before selecting `graph`. Dense and hybrid retrieval do not need it.

The builder uses [docling-graph](https://github.com/docling-project/docling-graph) 1.9.1 with a local Ollama model to extract biomedical entities (diseases, genes/proteins, chemicals/drugs) and their relations from each chunk. docling-graph is needed only to build the graph, not to query it, so it is an optional extra in the same environment:

```bash
uv sync --extra graph
```

The commands below run from the repository root on the host, like the *Local* variants above. The prebuilt index from `scripts/fetch_index.py` already includes the graph.

**Before you start**

- `data/bm25_corpus.jsonl` must exist. The ingestion step writes it. On the Docker path too, since Compose mounts `./data` into the container.

- Ollama must be running on the host, with the extractor model pulled:

  ```bash
  ollama pull llama3.2           # ~2 GB
  ```

The builder uses the Ollama address from `.env` (`OLLAMA_BASE_URL`), like the rest of the pipeline.

**1. Sample the chunks to extract**

```bash
uv run python graph/sample_chunks.py 300
```

This writes one Markdown file per chunk into `data/graph/sample/`, replacing any previous sample. Chunks are picked at an even stride across the whole corpus, with no reference to the benchmark queries, so the graph is not biased towards the questions it will be tested on. The shipped graph uses 300 chunks, and this command reproduces exactly that sample.

**2. Run the extraction**

```bash
uv run python graph/build_graph.py 8     # argument: concurrent requests (default 8)
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

In the dashboard, choose **graph** under *Retrieval Strategy* in the Live Monitor sidebar. The retriever loads the graph once and reloads it when the files change, but restart the dashboard so the Knowledge Base page picks up the new graph.

- **Local and Docker:** nothing else to do. Compose mounts `./data`, so the container sees the new `data/graph/` straight away.

> By default, graph results are fused with dense retrieval (`GRAPH_FUSE_DENSE=true`). While the graph covers only a small part of the corpus, fusion can reorder what dense retrieval found but can't bring in a passage that only the graph found. Set `GRAPH_FUSE_DENSE=false` to see graph-only results.

**Compared with LightRAG.** [graph/lightrag_results.md](graph/lightrag_results.md) compares this retriever with [LightRAG](https://github.com/HKUDS/LightRAG) on the same 50 chunks, extraction model and embedder:
- **Build:** this retriever's graph builds 3 times faster.
- **Query:** it answers in 5.8 ms (median), while LightRAG's graph modes take about 0.8 s, because they call the LLM on every query.

To rerun the comparison:

```bash
uv run --with lightrag-hku==1.5.7 --with opentelemetry-api==1.40.0 python graph/lightrag_compare.py build
uv run --with lightrag-hku==1.5.7 --with opentelemetry-api==1.40.0 python graph/lightrag_compare.py compare
```

---

## Docker Reference

### Service Overview

The `docker-compose.yml` defines two services:

| Service | Image | Port | Role |
|---|---|:---:|---|
| `ragscope` | `./docker/app/Dockerfile` | 8501 | Main application (pipeline + dashboard) |
| `chromadb` | `chromadb/chroma:1.5.9` | 8000 | Persistent vector store |

Where data lives:

| Storage | Holds | Removed by `docker compose down -v` |
|---|---|:---:|
| `chroma_data` (named volume) | The BioASQ and documents collections | Yes |
| `telemetry_data` (named volume) | Query telemetry records | Yes |
| `./data` (host folder) | BM25 corpus, knowledge graph, your documents | No |
| `./experiments/results` (host folder) | Benchmark CSVs and reports | No |

### Common Commands

```bash
# Start all services (detached)
docker compose up -d

# View logs for a specific service
docker compose logs -f ragscope

# Open a shell in the app container
docker compose exec ragscope bash

# Run a one-off Python command inside the container
docker compose exec ragscope python -c "from config.settings import settings; print(settings.ollama_base_url)"

# Rebuild and restart after code or dependency changes (watch mode does this
# for you; see below)
docker compose up -d --build

# Hard reset — removes all containers, networks, and volumes
docker compose down -v --remove-orphans
```

### Development Mode (Watch)

`docker-compose.yml` has a [Compose Watch](https://docs.docker.com/compose/how-tos/file-watch/) section. Start the stack with `--watch` to develop inside Docker without rebuilding by hand:

```bash
docker compose up --build --watch
```

- **Code edits** are copied into the running container within a couple of seconds, and Streamlit reloads the dashboard.
- **Changes to `pyproject.toml` or `uv.lock`** rebuild the image and restart the container, so new packages are always installed.

Without `--watch`, the container runs the code baked into its image.

The test suite runs on the host (`uv run pytest`), since the image is built without development dependencies.

### Dockerfile Summary

`docker/app/Dockerfile` (used by the `ragscope` service) is a two-stage build:

```dockerfile
# Stage 1 — builder: uv-managed dependency install (cached unless pyproject.toml / uv.lock change)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
RUN apt-get install -y build-essential gcc g++ zlib1g-dev   # native deps for some packages
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev           # deps only, no optional extras
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

> The `--frozen` flag ensures Docker builds are fully reproducible: the exact versions pinned in `uv.lock` are installed, and the build will fail if the lockfile is out of sync with `pyproject.toml`. The image does not install the optional `graph` extra (docling-graph), which is needed only to build the knowledge graph, not to query it.

---

## Evaluation Metrics

RAGScope computes the following metrics per query using the RAGAS framework (Es et al., 2024):

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

**Since submission.** Evaluating the platform further, in response to supervisory review, found the following:

- **Where the time goes:** ChromaDB accounts for under 0.1% of query time; the RAGAS judge and generation account for nearly all of it.
- **CPU-only use:** on a processor alone, the 3B models answer in about 25 s and finish evaluation in about 2 minutes.
- **LightRAG:** compared with this retriever, its graph modes are about 140 times slower per query, because they call the LLM on every query ([graph/lightrag_results.md](graph/lightrag_results.md)).

Full results and analysis:

- Chapter 4 of the dissertation (submitted September 2026)
- `experiments/results/`: the benchmark CSV, the analysis report and the figures. This folder is git-ignored; [Running the Benchmark Evaluation](#running-the-benchmark-evaluation) regenerates it.

---

## Limitations

The following limitations are acknowledged and discussed in full in the dissertation:

- **Domain specificity:** The evaluation corpus is BioASQ (biomedical abstracts); findings may not generalise to other domains (e.g., legal, general web text).
- **RAGAS judge bias:** RAGAS metrics are computed using an LLM judge, which may share biases with the evaluated model - particularly when the same model serves both roles. Mitigation: the judge model (Qwen 2.5) is held constant and separated from the evaluated LLM configurations where possible. The CPU profile offers `qwen2.5:3b` both as a generator and as the judge; choose `llama3.2:3b` as the generator to keep the two apart.
- **Hardware constraints:** The experiment ran on a single laptop (an Apple M5 MacBook Pro, 16 GB, with Ollama on its integrated GPU), which does not replicate production-scale latency profiles. Reported latency figures should be interpreted comparatively within experimental conditions, not as absolute production benchmarks.
- **Timeline constraints:** The 5.5-month dissertation window limits the scale of experimentation; a larger query benchmark would strengthen statistical conclusions.

Found since submission:

- **Approximate vector search:** ChromaDB's HNSW index is approximate, and the last 117 of the 30,850 vectors sit in ChromaDB's write-ahead log, which is re-inserted with several threads each time the store is opened. Which neighbours that produces is not fixed, so re-running retrieval reproduces the published top-5 passages for 199 or 200 of the 200 benchmark queries, depending on the run. This was observed by reopening the embedded store; the server replays the same log each time it restarts.

---

## Research Outputs

| Output | Status | Location |
|---|---|---|
| MSc Research Proposal | ✅ Complete | Submitted separately — not tracked in this repository |
| Literature Review | ✅ Complete | Submitted separately — not tracked in this repository |
| System Architecture | ✅ Complete | `docs/architecture.md`, `docs/images/ragscope_system_architecture.svg` |
| Platform Implementation | ✅ Complete | `pipeline/`, `evaluation/`, `dashboard/` |
| Benchmark Experiment | ✅ Complete | `experiments/results/` |
| LightRAG Baseline Comparison | ✅ Complete | `graph/lightrag_results.md` |
| Dissertation | ✅ Complete | — |
| Publication | 🔄 In progress | — |

---

## Citation

If you use RAGScope in your own research, please cite it as below, or use GitHub's **Cite this repository** button, which reads [`CITATION.cff`](CITATION.cff):

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
- Es et al. (2024) — RAGAS: Automated evaluation of RAG
- Guo et al. (2024) — LightRAG: Simple and fast retrieval-augmented generation
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