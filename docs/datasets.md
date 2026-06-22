# RAGScope — Datasets

> **MSc Data Science Dissertation · Leeds Beckett University · 2026**  
> Phase 3 Deliverable — Data Collection and Preparation

---

## Table of Contents

- [Overview](#overview)
- [Benchmark Composition](#benchmark-composition)
- [Dataset 1 — MS MARCO Passage Ranking](#dataset-1--ms-marco-passage-ranking)
- [Dataset 2 — Natural Questions](#dataset-2--natural-questions)
- [Dataset 3 — HotpotQA](#dataset-3--hotpotqa)
- [Data Preparation Pipeline](#data-preparation-pipeline)
  - [Text Cleaning](#text-cleaning)
  - [Token-Bounded Chunking](#token-bounded-chunking)
  - [Embedding Generation](#embedding-generation)
  - [Deduplication](#deduplication)
- [Ingestion Configuration](#ingestion-configuration)
- [Retrieval Corpus Composition](#retrieval-corpus-composition)
- [Benchmark Query Set Design](#benchmark-query-set-design)
- [Ground-Truth Alignment](#ground-truth-alignment)
- [Ethical and Licensing Considerations](#ethical-and-licensing-considerations)
- [Reproducibility](#reproducibility)
- [Loader API Reference](#loader-api-reference)
- [References](#references)

---

## Overview

Three publicly available, widely validated benchmark datasets are used in this research. Each dataset was selected to cover a distinct query type that tests a different aspect of RAG reliability and the observability platform's diagnostic capacity. Together they form a 200-query evaluation benchmark that is diverse in query structure, reasoning complexity, and grounding condition.

All datasets are:
- Available under open licences explicitly permitting academic re-use
- Hosted on HuggingFace Datasets Hub for reproducible, versioned downloading
- Well-documented in the peer-reviewed literature with established baselines
- Directly compatible with standard RAG evaluation tooling including RAGAS

**Selection rationale** (proposal §3 — Benchmark Corpus and Query Set):

> The evaluation corpus must test the platform across qualitatively different failure modes: simple factual retrieval where the answer is unambiguous, large-scale passage ranking where retrieval precision is the primary challenge, and multi-hop reasoning where the generative model must synthesise information across multiple passages — the hardest condition for faithfulness.

---

## Benchmark Composition

| Dataset | Query Type | Corpus Source | Queries | Passages in Retrieval Corpus | Licence |
|---|---|---|:---:|:---:|---|
| MS MARCO | Passage ranking | Web documents (Bing) | 60 | 50,000 sampled | MIT |
| Natural Questions | Single-hop QA | Wikipedia | 40 | ~40 supporting passages | CC BY-SA 3.0 |
| HotpotQA (fullwiki) | Multi-hop QA | Wikipedia (full corpus) | 100 | ~500–800 supporting passages | CC BY-SA 4.0 |
| **Total** | **Mixed** | | **200** | **~51,000** | |

The benchmark is **stratified** by query type to ensure the experimental results are not dominated by a single difficulty level. The 60/40/100 split was chosen to weight multi-hop queries more heavily because they represent the most demanding condition for RAG faithfulness and the most diagnostic scenario for the observability platform.

---

## Dataset 1 — MS MARCO Passage Ranking

### Overview

| Property | Value |
|---|---|
| Full name | Microsoft Machine Reading Comprehension |
| Version | v2.1 |
| Source | Real anonymised user queries submitted to the Bing search engine |
| Passages | ~8.8 million web-sourced passages |
| Training queries | ~1,000,000 |
| Development queries | 6,980 (with human-annotated relevance labels) |
| Licence | MIT |
| HuggingFace Hub | `ms_marco` / `v2.1` |
| Paper | Bajaj et al. (2016), arXiv:1611.09268 |
| Download | https://microsoft.github.io/msmarco/ |

### Characteristics

MS MARCO is the most widely used dataset for passage retrieval research and provides the primary large-scale retrieval corpus for this project. Its queries reflect naturalistic web search behaviour across diverse topics including factual lookup, how-to questions, and definitional queries.

**Why MS MARCO?**
- The 8.8M passage corpus provides realistic retrieval difficulty — relevant passages must be found among a large pool of distractors
- Human relevance annotations enable precise retrieval quality assessment
- Established baselines exist in the literature for direct comparison
- Open MIT licence permits unrestricted academic re-use

**Query types in the development set:**
- **Single-answer queries** — one passage in the corpus contains the answer; 30 sampled for the benchmark
- **Multi-passage queries** — multiple passages contain relevant information; 30 sampled for the benchmark

### Loader

```python
from data.loaders.msmarco import load_corpus, load_queries

# Load 50,000 passages for the retrieval corpus
passages = load_corpus(sample_size=50_000, seed=42)

# Load 60 stratified evaluation queries
queries = load_queries(sample_size=60, seed=42)
```

**Passage schema:**
```json
{
  "id":      "msmarco_<query_id>_<passage_idx>",
  "text":    "cleaned passage text",
  "title":   "",
  "dataset": "msmarco"
}
```

**Query schema:**
```json
{
  "id":         "query_id",
  "query":      "cleaned query text",
  "answers":    ["answer string"],
  "query_type": "single_answer | multi_passage",
  "dataset":    "msmarco"
}
```

### Usage in this Research

A stratified random sample of 50,000 passages is drawn from the training split and indexed into ChromaDB as the primary retrieval corpus. The sample is stratified by passage source domain to maintain diversity. The 60 evaluation queries are drawn from the development set (which has human-annotated relevance labels), split equally between single-answer (30) and multi-passage (30) types.

---

## Dataset 2 — Natural Questions

### Overview

| Property | Value |
|---|---|
| Full name | Natural Questions |
| Source | Real Google Search queries; answers from Wikipedia |
| Training set | 307,373 examples |
| Development set | 7,842 examples |
| Licence | CC BY-SA 3.0 |
| HuggingFace Hub | `natural_questions` |
| Paper | Kwiatkowski et al. (2019), TACL 7:452–466 |
| Download | https://ai.google.com/research/NaturalQuestions |

### Characteristics

Natural Questions (NQ) provides the single-hop factual QA component of the benchmark. Queries are real Google Search queries submitted by users, making them naturalistic and representative of genuine information-seeking behaviour — more authentic than artificially constructed test questions.

Each example includes:
- A real Google Search query
- A **long-answer** annotation — the Wikipedia passage containing the answer
- A **short-answer** annotation — the specific span within the passage

**Why Natural Questions?**
- Unambiguous short-answer ground truths enable precise RAGAS `answer_correctness` computation
- Naturalistic queries reflect realistic deployment conditions
- Wikipedia passages are cleanly structured with low noise
- The short-answer span serves as the reference answer for RAGAS evaluation

### Answer Extraction

NQ stores document content as flat token arrays with associated HTML markup flags. The loader (`data/loaders/natural_questions.py`) extracts long-answer passages by:

1. Reading the `long_answers` annotation for `start_token` and `end_token` indices
2. Slicing the `document.tokens.token` array
3. Filtering out tokens where `is_html == True` (removing markup)
4. Joining remaining tokens with spaces and applying the text cleaner

This produces a clean, HTML-free passage text. Examples where no extractable short answer or valid long answer exists are skipped.

### Loader

```python
from data.loaders.natural_questions import load_queries_and_passages

queries, passages = load_queries_and_passages(sample_size=40, seed=42)
```

**Query schema:**
```json
{
  "id":           "example_id",
  "query":        "What is the capital of France?",
  "short_answer": "Paris",
  "long_answer":  "Paris is the capital and most populous city of France...",
  "answers":      ["Paris"],
  "query_type":   "single_hop",
  "dataset":      "natural_questions"
}
```

**Supporting passage schema:**
```json
{
  "id":      "nq_<example_id>",
  "text":    "cleaned long-answer passage",
  "title":   "Wikipedia article title",
  "dataset": "natural_questions"
}
```

### Usage in this Research

40 queries are drawn from the development set, filtered to retain only examples with extractable short answers. The corresponding long-answer Wikipedia passages are added to the retrieval corpus. This ensures the ground-truth answer is always findable by the retriever — enabling a controlled measurement of faithfulness and correctness without retrieval failure confounding the results.

---

## Dataset 3 — HotpotQA

### Overview

| Property | Value |
|---|---|
| Full name | HotpotQA |
| Configuration | `fullwiki` (full Wikipedia retrieval setting) |
| Source | Crowd-sourced; Wikipedia articles |
| Development set | 7,405 examples |
| Licence | CC BY-SA 4.0 |
| HuggingFace Hub | `hotpot_qa` / `fullwiki` |
| Paper | Yang et al. (2018), EMNLP, pp. 2369–2380 |
| Download | https://hotpotqa.github.io/ |

### Characteristics

HotpotQA is the stress-test component of the benchmark. It is the hardest condition for RAG faithfulness because each question requires reasoning across **two or more Wikipedia articles** to arrive at the correct answer. The `fullwiki` setting does not provide the supporting documents — the retrieval component must identify them from the full Wikipedia corpus without guidance, making it a genuine open-domain multi-hop retrieval task.

**Question types:**
- **Bridge questions** — require chaining information sequentially across two entities. *Example: "What year was the director of [Film X] born?"* — requires finding the director of Film X, then finding their birth year.
- **Comparison questions** — require retrieving an attribute of two separate entities and comparing them. *Example: "Which country has a larger population, [Country A] or [Country B]?"*

**Why HotpotQA?**
- Multi-hop reasoning is the failure mode most strongly associated with hallucination in RAG systems (Barnett et al., 2024)
- The `fullwiki` setting tests retrieval under realistic open-domain conditions
- Bridge and comparison question types probe different multi-step reasoning patterns
- HotpotQA has extensive established baselines for retrieval and QA performance

### Supporting Documents

Each HotpotQA example includes a `context` field — a list of `[title, sentences]` pairs — which serves as the gold supporting documents. In the fullwiki setting, these documents are not provided at query time (the retriever must find them). However, RAGScope **adds these documents to the retrieval corpus** during ingestion so that:

1. The retriever has the opportunity to find the relevant passages
2. RAGAS faithfulness can be measured against the content of those passages
3. The experiment tests whether the retriever correctly surfaces the relevant supporting passages rather than distractors

### Loader

```python
from data.loaders.hotpotqa import load_queries_and_passages

queries, passages = load_queries_and_passages(sample_size=100, seed=42)
# Returns 50 bridge + 50 comparison queries
```

**Query schema:**
```json
{
  "id":               "5a8b57f25542992500000001",
  "query":            "Were Scott Derrickson and Ed Wood both directors?",
  "answer":           "yes",
  "answers":          ["yes"],
  "query_type":       "bridge",
  "supporting_facts": ["Scott Derrickson", "Ed Wood"],
  "passage_ids":      ["hotpotqa_<id>_Scott_Derrickson", "hotpotqa_<id>_Ed_Wood"],
  "dataset":          "hotpotqa"
}
```

**Supporting passage schema:**
```json
{
  "id":      "hotpotqa_<example_id>_<title_slug>",
  "text":    "Scott Derrickson is an American director...",
  "title":   "Scott Derrickson",
  "dataset": "hotpotqa"
}
```

### Usage in this Research

100 queries are drawn from the development set: 50 bridge and 50 comparison, stratified equally. Supporting passages from the `context` field of each example are added to the retrieval corpus. This produces a corpus where relevant passages exist and can in principle be retrieved — the experiment then measures whether they actually are retrieved and whether the LLM uses them faithfully.

---

## Data Preparation Pipeline

All three datasets pass through the same preparation pipeline before indexing. The pipeline is implemented in `data/preprocessing/` and orchestrated by `pipeline/ingestion.py`.

### Text Cleaning

**Module:** `data/preprocessing/cleaner.py`

Applied to every passage and query before any further processing.

| Step | Function | Purpose |
|---|---|---|
| Unicode normalisation | `_normalise_unicode()` | NFC normalisation for consistent accented character representation |
| HTML stripping | `_strip_html()` | Remove tags; decode common entities (`&amp;`, `&lt;`, etc.) |
| Control character removal | `_remove_control_chars()` | Strip ASCII 0x00–0x1F except newline/tab |
| Whitespace normalisation | `_normalise_whitespace()` | Collapse tabs, carriage returns, multiple spaces; limit blank lines |
| Punctuation collapsing | `_collapse_repeated_punctuation()` | `...` → `...`; `---` → `—`; `===` stripped |

**Meaningfulness filter:** `is_meaningful(text, min_chars=50)` — rejects passages shorter than 50 characters or where fewer than 40% of characters are alphabetic (e.g., navigation tables, code blocks). This prevents low-quality passages from degrading retrieval precision.

### Token-Bounded Chunking

**Module:** `data/preprocessing/chunker.py`

Passages are split into overlapping chunks using the `tiktoken` tokeniser (`cl100k_base` encoding — the same vocabulary as `all-MiniLM-L6-v2`). Splitting on token boundaries rather than character counts ensures chunks never exceed the embedding model's context window.

**Default configuration:**

| Parameter | Value | Rationale |
|---|---|---|
| `chunk_size` | 512 tokens | Fits comfortably within `all-MiniLM-L6-v2`'s 256-token limit after subword tokenisation differences; avoids truncation |
| `chunk_overlap` | 64 tokens | Preserves cross-boundary reasoning chains important for HotpotQA passages |

**Chunk schema (`Chunk` dataclass):**

| Field | Type | Description |
|---|---|---|
| `text` | str | Chunk text |
| `doc_id` | str | Source passage identifier |
| `chunk_index` | int | Zero-based position within the source passage |
| `start_token` | int | Token offset of first token in source |
| `end_token` | int | Token offset (exclusive) of last token |
| `metadata` | dict | `{"dataset": "...", "title": "..."}` |
| `chunk_id` | property | `{doc_id}::chunk_{chunk_index}` |

### Embedding Generation

**Module:** `pipeline/embeddings.py`  
**Model:** `sentence-transformers/all-MiniLM-L6-v2`

| Property | Value |
|---|---|
| Embedding dimension | 384 |
| Max sequence length | 256 tokens |
| Normalisation | L2-normalised (cosine similarity == dot product) |
| Device | Configurable: `cpu` / `cuda` / `mps` |
| Batch size | 64 (default) |

L2 normalisation is applied to all embeddings so that cosine similarity reduces to a simple dot product, which ChromaDB uses internally for efficient HNSW search.

### Deduplication

**Stage:** `pipeline/ingestion.py` → `_load_all_passages()`

Passage IDs are tracked in a `set[str]` across all three datasets. A passage is skipped if its ID has already been seen. This handles the case where HotpotQA examples share supporting passages (common for popular Wikipedia articles) and prevents:
- Redundant embeddings in ChromaDB consuming unnecessary storage
- Inflated BM25 index with duplicate entries that distort IDF scores
- Misleading ingestion statistics (counting the same passage twice)

Deduplication is reported in the ingestion manifest (`data/ingestion_manifest.json`) under `per_dataset[].passages_skipped`.

---

## Ingestion Configuration

The ingestion pipeline is fully configurable via CLI flags or environment variables. Key parameters affecting corpus characteristics:

```bash
uv run python pipeline/ingestion.py \
  --corpus all \           # Process all three datasets
  --chunk-size 512 \       # Tokens per chunk
  --chunk-overlap 64 \     # Token overlap
  --embed-batch-size 256 \ # Chunks per embedding forward pass
  --upsert-batch-size 500  # Chunks per ChromaDB upsert
```

**Reproducing the exact research corpus** (seed is set globally in `.env`):
```bash
# Ensure EXPERIMENT_RANDOM_SEED=42 in .env, then:
uv run python pipeline/ingestion.py --corpus all --reset
```

The `--reset` flag wipes the ChromaDB collection first, ensuring a clean reproducible state. Without `--reset`, ingestion is idempotent — running it a second time will upsert but not duplicate existing chunks.

---

## Retrieval Corpus Composition

After ingestion, the ChromaDB collection contains approximately:

| Source | Passages | Avg Chunks/Passage | Estimated Chunks |
|---|:---:|:---:|:---:|
| MS MARCO (sampled) | 50,000 | ~2.5 | ~125,000 |
| Natural Questions | ~40 | ~3.0 | ~120 |
| HotpotQA supporting passages | ~600 | ~2.5 | ~1,500 |
| **Total** | **~50,640** | | **~126,620** |

> **Note:** Actual chunk counts depend on passage lengths and overlap configuration. The ingestion manifest at `data/ingestion_manifest.json` reports exact counts for the specific run.

The BM25 index (`data/bm25_corpus.jsonl`) mirrors the ChromaDB collection exactly — one JSONL entry per chunk — ensuring consistent coverage between dense and hybrid retrieval.

---

## Benchmark Query Set Design

The 200-query benchmark is assembled by `evaluation/benchmark.py` → `load_benchmark()`:

```python
from evaluation.benchmark import load_benchmark

queries = load_benchmark(seed=42)
# Returns: 60 MS MARCO + 40 NQ + 100 HotpotQA = 200 queries, shuffled
```

**Stratification rationale:**

| Query type | Count | Reasoning |
|---|:---:|---|
| MS MARCO — single answer | 30 | Baseline: standard factual retrieval |
| MS MARCO — multi-passage | 30 | Tests retrieval across multiple relevant documents |
| NQ — single hop | 40 | Unambiguous ground truth for precise correctness measurement |
| HotpotQA — bridge | 50 | Sequential multi-hop: hardest faithfulness condition |
| HotpotQA — comparison | 50 | Attribute comparison: tests synthesis across entities |

All queries are shuffled with `random.Random(seed=42)` before being presented to the pipeline. Each condition in the 2×2 factorial experiment sees the same shuffled order.

---

## Ground-Truth Alignment

RAGAS `answer_correctness` requires a ground-truth reference answer. Alignment by dataset:

| Dataset | Ground Truth Source | Quality |
|---|---|---|
| MS MARCO | `answers` field from the dataset annotation | Human-authored; may have multiple valid answer strings |
| Natural Questions | `short_answer` token string extracted from the NQ annotation | Extracted verbatim from Wikipedia; highly precise |
| HotpotQA | `answer` field (free-form string, e.g. "yes", "1989", entity names) | Crowd-sourced; precise for factual questions |

For queries where the answer is a binary `"yes"` / `"no"` (HotpotQA comparison), RAGAS `answer_correctness` uses embedding-based semantic similarity — these short answers may score lower than expected even when the model answer is correct. This limitation is acknowledged in the dissertation's limitations section.

---

## Ethical and Licensing Considerations

### Data Licences

| Dataset | Licence | Restrictions |
|---|---|---|
| MS MARCO | MIT | No restrictions on academic use |
| Natural Questions | CC BY-SA 3.0 | Attribution required; derivative works must use same licence |
| HotpotQA | CC BY-SA 4.0 | Attribution required; derivative works must use same licence |

All three licences explicitly permit re-use in academic research. No modifications are made to the query-answer pairs themselves; only preprocessing (cleaning, chunking) is applied to the passage texts.

### Privacy

MS MARCO queries are derived from real Bing user searches that have been anonymised by Microsoft Research prior to public release. This project uses the passages and development-set queries only as a retrieval benchmark. No attempt is made to re-identify users, and the queries are treated strictly as evaluation artefacts.

Natural Questions queries are derived from real Google Search logs, similarly anonymised. HotpotQA questions are crowd-sourced by Amazon Mechanical Turk workers and contain no personally identifiable information.

### No Personal Data

No personal data is collected, stored, or processed at any stage of this research. The evaluation corpus contains only publicly released academic benchmark data.

---

## Reproducibility

The exact benchmark used in the dissertation is reproducible by setting:

```env
EXPERIMENT_RANDOM_SEED=42
```

and running:

```bash
# Step 1: Download all datasets
uv run python data/loaders/download_all.py

# Step 2: Ingest (wipe first for a clean state)
uv run python pipeline/ingestion.py --corpus all --reset

# Step 3: Load the benchmark
python -c "
from evaluation.benchmark import load_benchmark
qs = load_benchmark(seed=42)
print(f'{len(qs)} queries loaded')
print('First query:', qs[0]['query'][:80])
"
```

Dataset downloads are cached by HuggingFace Datasets in `data/raw/` and will not be re-downloaded on subsequent runs. The ingestion manifest at `data/ingestion_manifest.json` records the exact corpus composition and timestamps for each ingestion run.

---

## Loader API Reference

### `data/loaders/msmarco.py`

| Function | Returns | Description |
|---|---|---|
| `load_corpus(sample_size, seed)` | `list[dict]` | Random sample of passages for the retrieval corpus |
| `load_queries(sample_size, seed)` | `list[dict]` | Stratified sample of development-set queries |
| `stream_corpus()` | `Iterator[dict]` | Memory-efficient streaming alternative to `load_corpus` |

### `data/loaders/natural_questions.py`

| Function | Returns | Description |
|---|---|---|
| `load_queries_and_passages(sample_size, seed)` | `tuple[list[dict], list[dict]]` | Queries and their associated supporting passages |

### `data/loaders/hotpotqa.py`

| Function | Returns | Description |
|---|---|---|
| `load_queries_and_passages(sample_size, seed)` | `tuple[list[dict], list[dict]]` | Stratified bridge/comparison queries and supporting passages |
| `load_queries(sample_size, seed)` | `list[dict]` | Queries only (no passages) |

### `data/loaders/download_all.py`

One-shot download script. Calls all three loaders to pre-fetch and cache datasets:

```bash
uv run python data/loaders/download_all.py
```

Exits with code 1 if any download fails.

---

## References

- Bajaj, P. et al. (2016) 'MS MARCO: A human generated machine reading comprehension dataset', *arXiv:1611.09268*.
- Barnett, S. et al. (2024) 'Seven failure points when engineering a retrieval augmented generation system', *ICAIE 2024*.
- Kwiatkowski, T. et al. (2019) 'Natural Questions: A benchmark for question answering research', *TACL*, 7, pp. 452–466.
- Thakur, N. et al. (2021) 'BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models', *arXiv:2104.08663*.
- Yang, Z. et al. (2018) 'HotpotQA: A dataset for diverse, explainable multi-hop question answering', *EMNLP 2018*, pp. 2369–2380.