# RAGScope — Datasets

> **MSc Data Science Dissertation · Leeds Beckett University · 2026**
> Phase 3 Deliverable — Data Collection and Preparation

---

## Table of Contents

- [Overview](#overview)
- [Benchmark Composition](#benchmark-composition)
- [Dataset — BioASQ](#dataset--bioasq)
- [Role-Mapping Heuristic](#role-mapping-heuristic)
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

The evaluation benchmark is built from a single dataset, **BioASQ** — the large-scale biomedical semantic indexing and question-answering challenge — accessed via `rag-datasets/rag-mini-bioasq`, a HuggingFace mirror derived from the official BioASQ Task 11b training release. BioASQ was selected on the recommendation of the project supervisor: as a domain-specific biomedical benchmark, it tests the platform against realistic, expert-authored information needs rather than general-domain web or encyclopaedic text. The evaluation spans three conditions derived directly from BioASQ's own challenge structure — Phase A passage retrieval, and Phase B factoid/list and summary/yes-no question answering — each testing a qualitatively different aspect of RAG reliability.

The official BioASQ release (question-type labels, `exact_answer`/`ideal_answer` fields, RDF triples, and a ~23-million-abstract PubMed corpus) requires free registration at bioasq.org and is not a `pip install`-able dataset. `rag-mini-bioasq` provides a bounded, freely downloadable derivative — a 40,221-passage corpus and 4,719 question/answer pairs with relevance judgements — sufficient to reconstruct a dissertation-scale benchmark without a multi-million-document ingest. It carries **no native question-type field**, so the three evaluation roles below are reconstructed from answer shape and relevance-judgement count rather than an official annotation; see [Role-Mapping Heuristic](#role-mapping-heuristic).

**Selection rationale:**

> The evaluation corpus must test the platform across qualitatively different failure modes: simple factual retrieval where the answer is unambiguous, large-scale passage ranking where retrieval precision is the primary challenge, and multi-document reasoning where the generative model must synthesise information across multiple passages — the hardest condition for faithfulness. BioASQ's own task structure (Phase A retrieval; Phase B factoid/list; Phase B summary/yes-no) maps directly onto these three conditions within a single, domain-realistic corpus.

---

## Benchmark Composition

| Role | BioASQ Analogue | Corpus Source | Queries | Selection Criterion |
|---|---|---|:---:|---|
| Phase A | Passage retrieval | PubMed abstracts | 60 | Stratified by relevance-judgement count (single vs. multi-relevant) |
| Factoid | Factoid & List QA | PubMed abstracts | 40 | Short answer (≤ 6 words) |
| Summary | Summary & Yes/No QA | PubMed abstracts | 100 | Long-form or yes/no answer |
| **Total** | | | **200** | |

All 200 queries are drawn from a single, disjoint partition of the 4,719-row BioASQ QA pool (seed 42) — no question appears in more than one role. The retrieval corpus (27,972 passages after cleaning, from a raw 40,221) is shared across all three roles, since BioASQ Phase A/B retrieval in the real challenge operates over the same PubMed collection regardless of question type; see [Retrieval Corpus Composition](#retrieval-corpus-composition).

The benchmark is **stratified by role** to ensure experimental results are not dominated by a single difficulty level. The 60/40/100 split deliberately weights the hardest, most diagnostic condition (summary, 100 queries) more heavily than precise factual QA (factoid, 40) and passage retrieval (Phase A, 60).

---

## Dataset — BioASQ

### Overview

| Property | Value |
|---|---|
| Full name | BioASQ: Large-Scale Biomedical Semantic Indexing and Question Answering |
| Source | Official BioASQ Task 11b training release, mirrored via `rag-datasets/rag-mini-bioasq` |
| Corpus | PubMed abstracts — 40,221 passages (27,972 after cleaning; see [Text Cleaning](#text-cleaning)) |
| QA pairs | 4,719, each with a free-text answer and a list of relevant passage IDs |
| Licence | CC BY 2.5 |
| HuggingFace Hub | `rag-datasets/rag-mini-bioasq`, configs `text-corpus` and `question-answer-passages` |
| Official challenge | https://bioasq.org (registration required for the full release; not used here) |

### Characteristics

BioASQ questions are written by biomedical domain experts and answered against PubMed abstracts, spanning genetics, pharmacology, disease classification, and clinical research. Every query in the benchmark shares this single realistic domain, which is significant for evaluation: embedding models and LLMs are more likely to have systematic strengths or gaps concentrated in specialised vocabulary (gene names, drug names, clinical terminology) than in general web or encyclopaedic text.

**Why BioASQ?**
- Domain-expert-authored questions with genuine biomedical information needs, not synthetic or crowd-sourced approximations
- Relevance judgements (`relevant_passage_ids`) enable retrieval-quality stratification for the Phase A passage-retrieval role
- A single coherent domain lets the dissertation additionally comment on whether RAG failure modes identified on general-domain text (Chapter 2's literature review) transfer to a specialised domain
- Freely available without registration via `rag-mini-bioasq`, at a scale (tens of thousands of passages) compatible with a dissertation timeline

### Loader

```python
from data.loaders.bioasq import load_corpus, load_phase_a_queries, load_factoid_queries, load_summary_queries

# Load the full cleaned passage corpus (27,972 passages)
passages = load_corpus()

# Load the three query roles (60 + 40 + 100 = 200, mutually disjoint)
phase_a = load_phase_a_queries(sample_size=60, seed=42)
factoid = load_factoid_queries(sample_size=40, seed=42)
summary = load_summary_queries(sample_size=100, seed=42)
```

**Passage schema:**
```json
{
  "id":      "20598273",
  "text":    "cleaned PubMed abstract text",
  "title":   "",
  "dataset": "bioasq"
}
```

**Query schema (Phase A example):**
```json
{
  "id":                     "152",
  "query":                  "What is known about clinical efficacy of ceftriaxone for treatment of amyotrophic lateral sclerosis?",
  "answers":                ["There have been a few case reports to suggest that ceftriaxone can be effective..."],
  "relevant_passage_ids":   ["18326497", "22680643", "..."],
  "query_type":             "single_relevant | multi_relevant",
  "dataset":                "bioasq_phase_a"
}
```

Factoid and summary queries share the same shape, with `query_type` set to `"factoid"` or `"yesno" | "summary"` respectively, and `dataset` set to `"bioasq_factoid"` or `"bioasq_summary"`.

### Usage in this Research

The full cleaned corpus (27,972 passages) is indexed into ChromaDB — no subsampling is applied, since this is already a bounded, dissertation-appropriate scale. The 200 evaluation queries are partitioned from the 4,719-row QA pool into three disjoint roles by a single deterministic pass (`data/loaders/bioasq.py::_build_role_pools`), guaranteeing that no question is evaluated under more than one role even though all three roles draw from the same underlying pool.

---

## Role-Mapping Heuristic

`rag-mini-bioasq` does not preserve BioASQ's official question-type labels (factoid/list/summary/yesno), so each of the platform's three roles is reconstructed from the shape of the `answer` field and the size of `relevant_passage_ids`. Classification is applied in a fixed priority order over the full QA pool, so the three resulting pools are disjoint by construction:

1. **Factoid** (n = 40, sampled from 256 candidates) — answer ≤ 6 words. Approximates BioASQ's convention that "exact answers" are short entity/phrase strings (gene names, drug names, diagnoses).
2. **Summary — yes/no** (n = 50, sampled from 802 candidates) — remaining rows whose answer starts with "yes" or "no" (e.g. *"Yes, papilin is a secreted protein"*), standing in for BioASQ's yes/no question type.
3. **Summary — long-form** (n = 50, sampled from the first 50 of 3,661 remaining long-answer rows) — free-text paragraph answers requiring synthesis across multiple cited passages, standing in for BioASQ's summary question type.
4. **Phase A** (n = 60) — drawn from whatever remains *after* the summary role has claimed its 50 long-form rows (3,611 remaining), stratified by `len(relevant_passage_ids)`: 30 single-relevant (from 677 candidates) and 30 multi-relevant (from 2,934 candidates). This split is deliberately answer-shape-agnostic — Phase A in the real BioASQ challenge is a pure retrieval task, not an answer-extraction one.

This ordering is a real implementation detail, not just documentation: an earlier draft of the loader sampled Phase A and the summary role independently from the same overlapping long-answer pool, which could hand both roles the same question. The shipped implementation samples summary's share first and only then exposes the remainder to Phase A, which was verified empirically (zero ID overlap across all three role pools at seed 42) before being adopted.

**Threshold calibration.** The ≤ 6-word factoid threshold and the "yes"/"no" prefix rule were chosen after inspecting the real answer-length distribution (median 24 words; 5.4% of rows ≤ 6 words; 18.5% yes/no-prefixed), confirming enough candidates exist for each role at the target sample sizes before committing to the heuristic.

---

## Data Preparation Pipeline

The BioASQ corpus passes through a standard preparation pipeline before indexing, implemented in `data/preprocessing/` and orchestrated by `pipeline/ingestion.py`.

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

**Meaningfulness filter:** `is_meaningful(text, min_chars=50)` — rejects passages shorter than 50 characters or where fewer than 40% of characters are alphabetic. For BioASQ this filter is doing real work beyond its original navigation-table/code-block use case: of the 40,221 raw passages in `rag-mini-bioasq`'s `text-corpus` config, 12,220 (30.4%) are literal placeholder strings reading `"nan"` — a data-quality artefact of how this HuggingFace mirror was generated from the official release, confirmed by direct inspection, not a bug in this project's cleaning logic. The filter correctly discards them, leaving 27,972 real passages.

### Token-Bounded Chunking

**Module:** `data/preprocessing/chunker.py`

Passages are split into overlapping chunks using the `tiktoken` tokeniser (`cl100k_base` encoding — the same vocabulary as `all-MiniLM-L6-v2`). Splitting on token boundaries rather than character counts ensures chunks never exceed the embedding model's context window.

**Default configuration:**

| Parameter | Value | Rationale |
|---|---|---|
| `chunk_size` | 512 tokens | Fits comfortably within `all-MiniLM-L6-v2`'s 256-token limit after subword tokenisation differences; avoids truncation |
| `chunk_overlap` | 64 tokens | Preserves cross-boundary reasoning chains, important for BioASQ's summary-style passages |

**Chunk schema (`Chunk` dataclass):**

| Field | Type | Description |
|---|---|---|
| `text` | str | Chunk text |
| `doc_id` | str | Source passage identifier |
| `chunk_index` | int | Zero-based position within the source passage |
| `start_token` | int | Token offset of first token in source |
| `end_token` | int | Token offset (exclusive) of last token |
| `metadata` | dict | `{"dataset": "bioasq", "title": ""}` |
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

Passage IDs are tracked in a `set[str]`; a passage is skipped if its ID has already been seen. With a single source corpus this is a defensive guard rather than a load-bearing step — `rag-mini-bioasq`'s corpus rows already carry unique PubMed IDs — but the logic is kept general so the pipeline can add further corpora in future without changing this stage.

---

## Ingestion Configuration

The ingestion pipeline is fully configurable via CLI flags or environment variables. Key parameters affecting corpus characteristics:

```bash
uv run python pipeline/ingestion.py \
  --corpus bioasq \        # The single BioASQ corpus
  --chunk-size 512 \       # Tokens per chunk
  --chunk-overlap 64 \     # Token overlap
  --embed-batch-size 256 \ # Chunks per embedding forward pass
  --upsert-batch-size 500  # Chunks per ChromaDB upsert
```

**Reproducing the exact research corpus** (seed is set globally in `.env`):
```bash
# Ensure EXPERIMENT_RANDOM_SEED=42 in .env, then:
uv run python pipeline/ingestion.py --corpus bioasq --reset
```

The `--reset` flag wipes the ChromaDB collection first, ensuring a clean reproducible state. Without `--reset`, ingestion is idempotent — running it a second time will upsert but not duplicate existing chunks.

---

## Retrieval Corpus Composition

After ingestion, the ChromaDB collection contains the following (exact counts from `data/ingestion_manifest.json` for the research ingestion run, completed in 159.2 seconds):

| Source | Passages | Chunks | Avg Chunks/Passage |
|---|:---:|:---:|:---:|
| BioASQ | 27,972 | 30,850 | 1.10 |

> **Note:** BioASQ is indexed as a single, unified corpus with no per-source breakdown to report. The 1.10 average chunks/passage reflects the typical length of PubMed abstracts, with a modest share of longer abstracts splitting into two chunks. Chunk counts depend on passage lengths and the chunk-size/overlap configuration, so a differently configured or re-sampled ingestion run will not reproduce these exact figures — re-run `pipeline/ingestion.py` and consult the regenerated `data/ingestion_manifest.json` for that run's actual counts.

The BM25 index (`data/bm25_corpus.jsonl`) mirrors the ChromaDB collection exactly — one JSONL entry per chunk — ensuring consistent coverage between dense and hybrid retrieval.

---

## Benchmark Query Set Design

The 200-query benchmark is assembled by `evaluation/benchmark.py` → `load_benchmark()`:

```python
from evaluation.benchmark import load_benchmark

queries = load_benchmark(seed=42)
# Returns: 60 Phase A + 40 factoid + 100 summary = 200 queries, shuffled
```

**Stratification rationale:**

| Query type | Count | Reasoning |
|---|:---:|---|
| Phase A — single-relevant | 30 | Baseline: standard factual retrieval, one relevant passage |
| Phase A — multi-relevant | 30 | Tests retrieval across multiple relevant documents |
| Factoid | 40 | Unambiguous short ground truth for precise correctness measurement |
| Summary — yes/no | 50 | Binary judgement requiring synthesis across evidence |
| Summary — long-form | 50 | Multi-passage synthesis: hardest faithfulness condition |

All queries are shuffled with `random.Random(seed=42)` before being presented to the pipeline. Each condition in the 2×2 factorial experiment sees the same shuffled order.

---

## Ground-Truth Alignment

RAGAS `answer_correctness` requires a ground-truth reference answer. All three BioASQ roles draw their ground truth from the same `answer` field in the source QA pairs:

| Role | Ground Truth Source | Quality |
|---|---|---|
| Phase A | `answer` field, free-form (often multi-sentence) | Domain-expert-authored; may combine several relevant facts |
| Factoid | `answer` field, short (≤ 6 words) | Domain-expert-authored; entity/phrase-level precision |
| Summary | `answer` field, free-form or yes/no-prefixed | Domain-expert-authored; paragraph-length for synthesis questions |

For summary-role queries with a yes/no-prefixed answer (e.g. *"Yes, mutations in the DNA that affect the splicing pattern of genes have been linked..."*), RAGAS `answer_correctness` uses embedding-based semantic similarity against the full sentence, not just the leading "yes"/"no" token — a known limitation of embedding-based correctness scoring for short binary-leaning answers, acknowledged here for BioASQ's yes/no questions.

---

## Ethical and Licensing Considerations

### Data Licence

| Dataset | Licence | Restrictions |
|---|---|---|
| BioASQ (via `rag-mini-bioasq`) | CC BY 2.5 | Attribution required |

The licence explicitly permits re-use in academic research. No modifications are made to the query-answer pairs themselves; only preprocessing (cleaning, chunking) is applied to the passage texts.

### Privacy

BioASQ questions are authored by biomedical domain experts as part of the official BioASQ challenge, not derived from real user search logs. The underlying passages are PubMed abstracts — published, publicly available scientific literature. No personally identifiable information is present in either the questions or the corpus.

### No Personal Data

No personal data is collected, stored, or processed at any stage of this research. The evaluation corpus contains only publicly released academic benchmark data and published biomedical abstracts.

---

## Reproducibility

The exact benchmark used in the dissertation is reproducible by setting:

```env
EXPERIMENT_RANDOM_SEED=42
```

and running:

```bash
# Step 1: Download BioASQ
uv run python data/loaders/download_all.py

# Step 2: Ingest (wipe first for a clean state)
uv run python pipeline/ingestion.py --corpus bioasq --reset

# Step 3: Load the benchmark
python -c "
from evaluation.benchmark import load_benchmark
qs = load_benchmark(seed=42)
print(f'{len(qs)} queries loaded')
print('First query:', qs[0]['query'][:80])
"
```

Dataset downloads are cached by HuggingFace Datasets in `data/raw/bioasq/` and will not be re-downloaded on subsequent runs. The ingestion manifest at `data/ingestion_manifest.json` records the exact corpus composition and timestamps for each ingestion run.

---

## Loader API Reference

### `data/loaders/bioasq.py`

| Function | Returns | Description |
|---|---|---|
| `load_corpus(seed=None)` | `list[dict]` | Full cleaned passage corpus (27,972 passages; no subsampling) |
| `load_phase_a_queries(sample_size=60, seed=None)` | `list[dict]` | Phase A role, stratified single-/multi-relevant |
| `load_factoid_queries(sample_size=40, seed=None)` | `list[dict]` | Factoid role, short-answer rows |
| `load_summary_queries(sample_size=100, seed=None)` | `list[dict]` | Summary role, stratified yes/no and long-form |

All three query functions call the same internal `_build_role_pools(seed)` partition, so calling any two with the same seed is guaranteed never to return overlapping queries.

### `data/loaders/download_all.py`

One-shot download script. Fetches both `rag-mini-bioasq` configs (`text-corpus` and `question-answer-passages`) to pre-fetch and cache the dataset:

```bash
uv run python data/loaders/download_all.py
```

Exits with code 1 if the download fails.

---

## References

- Nentidis, A., Katsimpras, G., Krithara, A., Lima-López, S., Farré-Maduell, E., Krallinger, M., Loukachevitch, N., Davydova, V., Tutubalina, E. and Paliouras, G. (2024) 'Overview of BioASQ 2024: The twelfth BioASQ challenge on Large-Scale Biomedical Semantic Indexing and Question Answering', *arXiv preprint arXiv:2508.20532*.
- Barnett, S. et al. (2024) 'Seven failure points when engineering a retrieval augmented generation system', *ICAIE 2024*.
- `rag-datasets/rag-mini-bioasq` — HuggingFace Datasets Hub, derived from the official BioASQ Task 11b training release. https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq
