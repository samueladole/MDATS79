# RAGScope — Evaluation Metrics

> **MSc Data Science Dissertation · Leeds Beckett University · 2026**  
> Phase 6 Deliverable — Evaluation Metrics Implementation

---

## Table of Contents

- [Overview](#overview)
- [RAGAS Evaluation Framework](#ragas-evaluation-framework)
  - [Context Relevance](#context-relevance)
  - [Answer Faithfulness](#answer-faithfulness)
  - [Answer Correctness](#answer-correctness)
  - [RAGAS Judge Model](#ragas-judge-model)
  - [Limitations of RAGAS](#limitations-of-ragas)
- [Hallucination Risk Score](#hallucination-risk-score)
  - [Formula and Rationale](#formula-and-rationale)
  - [Risk Bands](#risk-bands)
  - [Interpretation Guide](#interpretation-guide)
- [Telemetry Metrics](#telemetry-metrics)
  - [Latency Metrics](#latency-metrics)
  - [Token Metrics](#token-metrics)
  - [Cost Estimation](#cost-estimation)
- [Statistical Analysis Methods](#statistical-analysis-methods)
  - [Descriptive Statistics](#descriptive-statistics)
  - [Effect Sizes — Cohen's d](#effect-sizes--cohens-d)
  - [Pearson Correlation](#pearson-correlation)
  - [Pairwise Comparisons](#pairwise-comparisons)
- [Metric Interactions and Trade-offs](#metric-interactions-and-trade-offs)
- [Implementation Reference](#implementation-reference)
  - [RAGASRunner](#ragasrunner)
  - [HallucinationScore](#hallucinationscore)
  - [Metrics Module](#metrics-module)
  - [TokenCounter](#tokencounter)
- [Worked Example](#worked-example)
- [Known Limitations](#known-limitations)
- [References](#references)

---

## Overview

RAGScope computes two categories of metrics for every query execution:

**Quality metrics** — computed via the RAGAS framework using an auxiliary LLM judge, measuring how well the generated answer is supported by the retrieved context and how accurately it addresses the query.

**Telemetry metrics** — collected inline during pipeline execution using wall-clock timers and the Ollama API's reported token counts, measuring operational performance without any auxiliary model.

Together these six metrics form the dependent variable set for the 2×2 factorial experiment (Phase 8):

| Metric | Category | Range | Lower is better? |
|---|---|:---:|:---:|
| Context Relevance | Quality | [0, 1] | No |
| Answer Faithfulness | Quality | [0, 1] | No |
| Answer Correctness | Quality | [0, 1] | No |
| Hallucination Risk | Quality (composite) | [0, 1] | Yes |
| End-to-End Latency | Telemetry | ms | Yes |
| Total Tokens | Telemetry | count | Yes |

---

## RAGAS Evaluation Framework

**Reference:** Es et al. (2023), arXiv:2309.15217

RAGAS (Retrieval-Augmented Generation Assessment) provides automated, reference-free evaluation of RAG systems using an auxiliary LLM to judge the quality of query-answer-context triples. All RAGAS metrics are computed per-query and stored in the telemetry record.

RAGScope uses **RAGAS ≥ 0.2** via its `EvaluationDataset` API. Each query is wrapped in a `SingleTurnSample` and evaluated against the selected metrics using the configured judge model.

### Context Relevance

**RAGAS metric name:** `context_relevancy`  
**Range:** [0, 1]  
**Higher = better**

**Definition:** The proportion of the retrieved context that is relevant to the query. Specifically, RAGAS estimates how much of the retrieved passage content is actually needed to answer the question, versus being distracting or irrelevant.

**How it is computed (RAGAS internal):**
1. The judge LLM identifies sentences or segments in the retrieved context that are relevant to the query.
2. The proportion of relevant sentences to total sentences is computed as the score.

**Interpretation:**

| Score Range | Interpretation |
|:---:|---|
| 0.80 – 1.00 | Retrieved context is highly focused; retriever is working well |
| 0.60 – 0.79 | Most context is relevant; some noise present |
| 0.40 – 0.59 | Significant retrieval noise; answer quality is at risk |
| 0.00 – 0.39 | Retrieval has largely failed; returned passages are mostly irrelevant |

**What a low score indicates:** The retrieval component returned passages that are topically adjacent but not directly useful for answering the query. In the observability dashboard this is surfaced as an orange or red context relevance card — a signal to inspect the retrieved chunks in the Query Explorer.

**Relationship to hallucination:** Low context relevance is the *upstream cause* of hallucination. If the context is irrelevant, the LLM cannot be faithful to it, forcing reliance on parametric (training-data) knowledge — the primary hallucination pathway in RAG systems (Barnett et al., 2024).

---

### Answer Faithfulness

**RAGAS metric name:** `answer_faithfulness`  
**Range:** [0, 1]  
**Higher = better**

**Definition:** The degree to which every claim in the generated answer is grounded in the retrieved context. A faithful answer makes only claims that can be directly verified against the retrieved passages.

**How it is computed (RAGAS internal):**
1. The judge LLM decomposes the generated answer into individual atomic claims.
2. Each claim is evaluated against the retrieved context to determine whether it is supported.
3. The faithfulness score is the fraction of claims that are supported: `supported_claims / total_claims`.

**Interpretation:**

| Score Range | Interpretation |
|:---:|---|
| 0.85 – 1.00 | Answer is almost entirely grounded in the context |
| 0.65 – 0.84 | Mostly faithful; a small number of unsupported claims |
| 0.40 – 0.64 | Significant unfaithfulness; multiple claims may be hallucinated |
| 0.00 – 0.39 | Answer is largely hallucinated; context has been ignored or misrepresented |

**What a low score indicates:** The LLM has either ignored the retrieved context or generated claims beyond what the context supports. This is the direct signal of hallucination in RAG output, and it is the primary metric weighted in the composite hallucination risk score.

**Note on binary questions:** For BioASQ summary-role yes/no questions, faithfulness scoring may behave unexpectedly because atomic claim decomposition is not well-defined for short, binary-prefixed answers (e.g. "Yes, papilin is a secreted protein"). Scores for this query type should be interpreted with caution.

---

### Answer Correctness

**RAGAS metric name:** `answer_correctness`  
**Range:** [0, 1]  
**Higher = better**  
**Requires ground truth**

**Definition:** The semantic similarity between the generated answer and the provided ground-truth reference answer. This combines both factual overlap (did the model state the correct facts?) and semantic equivalence (did it express them accurately?).

**How it is computed (RAGAS internal):**
1. **Factual similarity** — the judge LLM identifies factual claims in both the generated answer and the ground truth, computing a weighted F1-style overlap score.
2. **Semantic similarity** — both answers are embedded; cosine similarity between their embedding vectors is computed.
3. The final score is a weighted combination of factual and semantic similarity.

**Ground-truth sources by role:**

| Role | Ground Truth | Notes |
|---|---|---|
| Phase A | `answer` field | Domain-expert-authored; often multi-sentence |
| Factoid | `answer` field (≤ 6 words) | Domain-expert-authored; entity/phrase-level precision |
| Summary | `answer` field | Free-text or "yes"/"no"-prefixed; precise for factual questions |

**When ground truth is absent:** If no `ground_truth` is supplied to `RAGPipeline.query()`, `answer_correctness` is not computed and the telemetry record stores `None`. The hallucination risk score is still computed from faithfulness and context relevance alone.

**Interpretation:**

| Score Range | Interpretation |
|:---:|---|
| 0.80 – 1.00 | Answer is semantically equivalent to the reference; factually correct |
| 0.60 – 0.79 | Largely correct; some missing details or minor factual errors |
| 0.40 – 0.59 | Partially correct; significant factual differences from reference |
| 0.00 – 0.39 | Substantially incorrect; answer diverges significantly from ground truth |

---

### RAGAS Judge Model

The auxiliary LLM used for all RAGAS metric computations is configured by `RAGAS_JUDGE_MODEL` (default: `qwen2.5:7b`).

**Critical methodological control:** The judge model is held **constant across all four experimental conditions** (Conditions A, B, C, D). This ensures that differences in RAGAS scores between conditions reflect genuine differences in RAG pipeline performance, not differences in judge behaviour.

```
Condition A: Llama3 + Dense   →  judge: qwen2.5:7b (constant)
Condition B: Llama3 + Hybrid  →  judge: qwen2.5:7b (constant)
Condition C: Mistral + Dense  →  judge: qwen2.5:7b (constant)
Condition D: Mistral + Hybrid →  judge: qwen2.5:7b (constant)
```

**Implementation:** RAGAS is configured to use `langchain_ollama.ChatOllama` for judge calls, routed through the local Ollama service — no external API calls are made. This client is rebuilt fresh on every `evaluate()` call rather than cached: `ragas.evaluate()` runs each call inside its own `asyncio.run(...)` (a new event loop per call), and a `ChatOllama` instance's async HTTP connections stay bound to whichever loop first used them, so a cached instance shared across calls fails with `RuntimeError: Event loop is closed` from the second call onward.

Embedding comparisons for `answer_correctness` reuse the pipeline's own `all-MiniLM-L6-v2` model (`pipeline/embeddings.py`) via a small `Embeddings` adapter, **not** `langchain_ollama.OllamaEmbeddings` against the judge model — `qwen2.5:7b` is a chat model, and Ollama's embeddings endpoint rejects a model that wasn't loaded in embedding-serving mode (`This server does not support embeddings. Start it with --embeddings`). Unlike the judge LLM, this embeddings adapter is synchronous and holds no event-loop-bound resources, so it's built once and reused across calls.

```python
from evaluation.ragas_runner import RAGASRunner

runner = RAGASRunner(judge_model="llama3")
scores = runner.evaluate(
    query="What is RAG?",
    answer="RAG combines LLMs with external retrieval...",
    contexts=["Retrieval-Augmented Generation is a technique..."],
    ground_truth="Retrieval-Augmented Generation",
)
# scores = {"context_relevance": 0.82, "answer_faithfulness": 0.91, "answer_correctness": 0.78}
```

---

### Limitations of RAGAS

The following limitations are acknowledged in the dissertation and should be considered when interpreting results:

**Judge model bias:** RAGAS metrics are computed using an LLM judge (Qwen2.5 7B), which may carry its own systematic biases regardless of which generator produced the answer. Qwen2.5 is deliberately a third model distinct from both generators (Llama 3, Mistral 7B), avoiding generator–judge overlap, but this does not eliminate judge-level bias entirely — it was not cross-validated against human annotations within the dissertation timeline.

**Short-answer edge cases:** For queries with very short ground-truth answers (e.g., "yes", "no", entity names of 1–2 words), the atomic claim decomposition step of faithfulness scoring and the factual overlap step of correctness scoring may behave unpredictably. Scores for BioASQ summary-role yes/no questions should be interpreted at the group level (mean across 50 queries) rather than individually.

**Reference-free vs. reference-based:** Context relevance and faithfulness are reference-free — they do not require ground-truth answers. Answer correctness is reference-based and therefore only meaningful when a high-quality ground truth is available. The quality of ground-truth annotations varies across the three benchmark datasets.

**Computational cost:** RAGAS metric computation adds substantial latency per query — observed in the tens of seconds during development, depending on judge model speed and context length, because each metric requires at least one additional LLM call. This latency is measured separately as `evaluation_ms` (not folded into `generation_ms`, which is generation-only), but it **is** included in `e2e_ms` and often dominates it — evaluation is real pipeline latency from an operational standpoint, even though it's conceptually a distinct concern from generation. See [Latency Metrics](#latency-metrics) for the full breakdown.

---

## Hallucination Risk Score

**Module:** `evaluation/hallucination_score.py`

### Formula and Rationale

```
hallucination_risk = 1 − (0.6 × answer_faithfulness + 0.4 × context_relevance)
```

This composite score is derived from the two RAGAS metrics most directly connected to hallucination. The design follows the analytical reasoning in the research proposal:

**Why faithfulness weight = 0.6:**  
Answer faithfulness directly measures whether the model's output is grounded in the retrieved context. An unfaithful answer — one containing claims not supported by the context — is the definitional hallucination in a RAG system (Ji et al., 2023). It is therefore the primary component of the risk score.

**Why context relevance weight = 0.4:**  
Context relevance is the upstream causal factor. If the retriever returns irrelevant context, the generative model cannot produce a faithful answer regardless of its own quality. Low context relevance therefore increases hallucination risk indirectly, by providing the model with nothing reliable to be faithful to (Barnett et al., 2024). It receives secondary weighting.

**Why the score is inverted (1 − ...):**  
The composite is inverted so that **higher score = higher risk**, matching the intuitive reading of a risk gauge on the dashboard. A score of 0.0 means perfect faithfulness and context relevance (zero risk). A score of 1.0 means complete unfaithfulness and irrelevant context (maximum risk).

**Example calculations:**

| Faithfulness | Context Relevance | Calculation | Risk Score | Band |
|:---:|:---:|---|:---:|---|
| 1.00 | 1.00 | 1 − (0.6×1.00 + 0.4×1.00) | **0.00** | LOW |
| 0.80 | 0.90 | 1 − (0.6×0.80 + 0.4×0.90) | **0.16** | LOW |
| 0.60 | 0.70 | 1 − (0.6×0.60 + 0.4×0.70) | **0.36** | MEDIUM |
| 0.40 | 0.50 | 1 − (0.6×0.40 + 0.4×0.50) | **0.56** | MEDIUM |
| 0.20 | 0.30 | 1 − (0.6×0.20 + 0.4×0.30) | **0.76** | HIGH |
| 0.00 | 0.00 | 1 − (0.6×0.00 + 0.4×0.00) | **1.00** | HIGH |

**Edge case — missing inputs:** If either faithfulness or context relevance is `None` (i.e., RAGAS failed to compute the metric), `compute_hallucination_risk()` returns `None` rather than producing a misleading partial score. The dashboard displays "UNKNOWN" for these cases.

**Clamping:** The result is clamped to [0, 1] to handle floating-point edge cases where the formula could produce a value infinitesimally outside the valid range.

---

### Risk Bands

| Band | Score Range | Colour | Meaning |
|---|:---:|---|---|
| **LOW** | [0.00, 0.35) | `#2ecc71` (green) | Answer is likely faithful and well-grounded; safe to use |
| **MEDIUM** | [0.35, 0.65) | `#f39c12` (amber) | Some hallucination risk present; review retrieved context |
| **HIGH** | [0.65, 1.00] | `#e74c3c` (red) | Likely hallucinated; do not rely on this answer |
| **UNKNOWN** | `None` | `#95a5a6` (grey) | RAGAS evaluation failed or ground truth unavailable |

Band thresholds are defined as constants in `evaluation/hallucination_score.py`:
```python
LOW_THRESHOLD    = 0.35
MEDIUM_THRESHOLD = 0.65
```

### Interpretation Guide

**Using the risk score in practice:**

A HIGH risk score does not necessarily mean every claim is wrong. It means the answer contains claims that cannot be verified against the retrieved context — those claims may still be factually accurate if the model is drawing on correct parametric knowledge. However, in a RAG system the whole point of retrieval is to constrain the model to verifiable knowledge; a HIGH score indicates this constraint has failed.

**Using the risk score for system diagnostics:**

The risk score is most useful when examined alongside the two component metrics:

| Pattern | Diagnosis | Recommended action |
|---|---|---|
| Low faithfulness + Low context relevance | Retrieval failure cascaded into generation failure | Check retrieval strategy; inspect top-k chunks in Query Explorer |
| Low faithfulness + High context relevance | Generation failure despite good retrieval | LLM is ignoring or misrepresenting relevant context; consider a different model or prompt |
| High faithfulness + Low context relevance | Model is being faithful to irrelevant context | Retrieval is returning wrong passages; review chunking or query formulation |
| High faithfulness + High context relevance | LOW risk; system working correctly | — |

---

## Telemetry Metrics

Telemetry metrics are collected inline during pipeline execution — no auxiliary model or post-processing is required.

### Latency Metrics

All latency measurements use Python's `time.perf_counter()`, which provides sub-millisecond wall-clock resolution on all supported platforms.

| Metric | Field name | Unit | Description |
|---|---|---|---|
| Embed query latency | `embed_query_ms` | ms | Time to encode the user query into a 384d vector |
| Vector search latency | `vector_search_ms` | ms | ChromaDB HNSW approximate nearest-neighbour search — **dense retrieval only**; 0 for hybrid |
| Dense candidate search latency | `dense_search_ms` | ms | ChromaDB candidate search prior to fusion — **hybrid retrieval only**; 0 for dense |
| BM25 search latency | `bm25_search_ms` | ms | BM25Okapi scoring over the in-memory corpus — **hybrid retrieval only**; 0 for dense |
| RRF fusion latency | `rrf_fusion_ms` | ms | Reciprocal Rank Fusion merge operation — **hybrid retrieval only**; 0 for dense |
| Total retrieval latency | `retrieval_ms` | ms | Sum of the applicable retrieval substeps above |
| Generation latency | `generation_ms` | ms | Ollama `/api/generate` wall-clock time (includes prompt processing + token generation) |
| Evaluation latency | `evaluation_ms` | ms | Wall-clock time inside the RAGAS `evaluate()` call; 0 if evaluation was skipped |
| End-to-end latency | `e2e_ms` | ms | Total pipeline latency — retrieval + generation + evaluation — taken from the `Timer` wrapping the whole of `RAGPipeline.query()` |

**Important:** `generation_ms` is reported by measuring wall-clock time around the `httpx` POST call to Ollama's `/api/generate` endpoint. It includes:
- Prompt tokenisation by the model
- KV cache computation
- Autoregressive token generation
- HTTP round-trip overhead

It does **not** include RAGAS evaluation time — that is measured separately as `evaluation_ms`, which is **included in `e2e_ms`**. In practice, evaluation latency routinely dominates total latency: it requires one or more auxiliary judge-LLM calls (context relevance, faithfulness, and optionally answer correctness), each comparable in cost to the generation call itself. A query with ~5s of generation time observed during development took ~36s of evaluation time — evaluation was the majority of `e2e_ms`, not a rounding error. Earlier versions of this pipeline computed `e2e_ms` as `retrieval_ms + generation_ms` only, silently dropping evaluation time from the persisted telemetry record; this has been corrected so `e2e_ms` reflects true wall-clock latency.

**Dense retrieval latency breakdown:**
```
retrieval_ms = embed_query_ms + vector_search_ms
```

**Hybrid retrieval latency breakdown:**
```
retrieval_ms = embed_query_ms + dense_search_ms + bm25_search_ms + rrf_fusion_ms
```

**Full end-to-end breakdown:**
```
e2e_ms = retrieval_ms + generation_ms + evaluation_ms + negligible overhead
         (token counting, telemetry serialisation — typically <5ms)
```

### Token Metrics

Token counts are reported directly by Ollama's API in the `/api/generate` response:

| Field | Ollama API field | Description |
|---|---|---|
| `prompt_tokens` | `prompt_eval_count` | Tokens in the full input prompt (query + all context passages) |
| `completion_tokens` | `eval_count` | Tokens generated in the response |
| `total_tokens` | Derived sum | `prompt_tokens + completion_tokens` |

**Fallback counting:** `telemetry/token_counter.py` provides `count_prompt()` and `count_tokens()` via tiktoken (`cl100k_base`) as a fallback if Ollama returns zero counts. In practice, Ollama consistently returns accurate counts for Llama 3 and Mistral.

**Token count implications for cost:**  
Prompt token counts grow with `top_k` and chunk size — more retrieved context means a longer prompt. This creates a direct trade-off: higher `top_k` improves retrieval recall but increases prompt cost. This trade-off is explicitly analysed in the experimental results via the Pearson correlation between `total_tokens` and quality metrics.

### Cost Estimation

Cost is estimated from the configurable per-token rate table:

```
estimated_cost_usd = (prompt_tokens / 1000) × COST_PER_1K_PROMPT_TOKENS
                   + (completion_tokens / 1000) × COST_PER_1K_COMPLETION_TOKENS
```

For local Ollama inference (the research default), both rates are `$0.00` and all costs are `$0.000000`. The cost field is included so that the same platform can estimate costs for cloud API comparisons (e.g., a follow-on study comparing Llama 3 local vs. GPT-4 API) without code changes.

---

## Statistical Analysis Methods

The experiment analysis (`experiments/analyse_results.py`) applies three statistical methods to the benchmark results, following the analysis plan in the research proposal.

All statistical functions are implemented in pure Python in `evaluation/metrics.py` — no numpy or scipy dependency — to keep the analysis reproducible without the full ML stack and to document the formulae explicitly.

### Descriptive Statistics

Computed for each metric across each of the four experimental conditions.

**Statistics reported:**

| Statistic | Description |
|---|---|
| n | Number of valid (non-null) values |
| Mean | Arithmetic mean |
| Std | Sample standard deviation (Bessel's correction: n−1 denominator) |
| Median | 50th percentile via linear interpolation |
| Q1 | 25th percentile |
| Q3 | 75th percentile |
| IQR | Interquartile range (Q3 − Q1) |
| Min | Minimum value |
| Max | Maximum value |

**Missing value handling:** `None` and `float('nan')` values are excluded before computing any statistic. The `n` field reports the count of valid values, so the analyst can assess how many queries produced valid metric scores.

**Percentile computation** uses linear interpolation:
```
index = (p/100) × (n − 1)
result = sorted_values[floor(index)] × (1 − frac) + sorted_values[ceil(index)] × frac
where frac = index − floor(index)
```

### Effect Sizes — Cohen's d

Cohen's d quantifies the practical significance of differences between two conditions, independently of sample size. Statistical significance (p-values) alone is insufficient for a dissertation-level analysis because even trivial differences can reach significance with 200 queries per condition.

**Formula (pooled standard deviation):**

```
d = (mean_A − mean_B) / pooled_SD

where:
    pooled_SD = sqrt(((n_A − 1) × var_A + (n_B − 1) × var_B) / (n_A + n_B − 2))
```

**Magnitude interpretation** (Cohen, 1988):

| |d| | Magnitude | Practical significance |
|:---:|---|---|
| < 0.20 | Negligible | Difference is smaller than typical measurement noise |
| 0.20 – 0.49 | Small | Detectable but modest practical difference |
| 0.50 – 0.79 | Medium | Noticeable practical difference |
| ≥ 0.80 | Large | Substantial practical difference |

**Sign convention:** `d` is positive when Condition A has a higher mean than Condition B. For quality metrics (higher = better), a positive d means Condition A performs better. For `hallucination_risk` and `e2e_ms` (lower = better), a positive d means Condition A performs *worse*.

**Usage example:**
```python
from evaluation.metrics import cohens_d

result = cohens_d(
    group_a=[0.82, 0.79, 0.85, ...],   # Llama3 + Dense faithfulness scores
    group_b=[0.75, 0.71, 0.78, ...],   # Mistral + Dense faithfulness scores
    label_a="Llama3+Dense",
    label_b="Mistral+Dense",
    metric="answer_faithfulness",
)
print(result.d)          # e.g., 0.42
print(result.magnitude)  # "small"
```

### Pearson Correlation

Pearson's r characterises the linear relationship between pairs of metrics, specifically used to address **RQ3**: what trade-offs exist between latency and quality?

**Formula:**
```
r = Σ[(xᵢ − x̄)(yᵢ − ȳ)] / sqrt(Σ(xᵢ − x̄)² × Σ(yᵢ − ȳ)²)
```

**Interpretation:**

| r | Interpretation |
|:---:|---|
| 0.70 – 1.00 | Strong positive correlation |
| 0.30 – 0.69 | Moderate positive correlation |
| 0.00 – 0.29 | Weak positive correlation |
| −0.29 – 0.00 | Weak negative correlation |
| −0.69 – −0.30 | Moderate negative correlation |
| −1.00 – −0.70 | Strong negative correlation |

**Pairs analysed:**

| Pair | Research question |
|---|---|
| `e2e_ms` vs `context_relevance` | Does higher latency (more thorough retrieval) improve context quality? |
| `e2e_ms` vs `answer_faithfulness` | Is there a latency–faithfulness trade-off? |
| `e2e_ms` vs `answer_correctness` | Does slower generation produce more accurate answers? |
| `e2e_ms` vs `hallucination_risk` | Does higher latency reduce hallucination risk? |

**Missing value handling:** pairs where either `x` or `y` is `None` are excluded. `pearson_r()` returns `None` if fewer than 3 valid paired values exist.

### Pairwise Comparisons

The analysis reports Cohen's d for four pairwise comparisons:

| Comparison | Addresses |
|---|---|
| Llama3+Dense vs Llama3+Hybrid | Effect of retrieval strategy (holding LLM constant) for Llama 3 |
| Mistral+Dense vs Mistral+Hybrid | Effect of retrieval strategy for Mistral |
| Llama3+Dense vs Mistral+Dense | Effect of LLM choice (holding retrieval constant) for Dense |
| Llama3+Hybrid vs Mistral+Hybrid | Effect of LLM choice for Hybrid |

These four comparisons decompose the 2×2 interaction into its main effects, directly addressing RQ3.

---

## Metric Interactions and Trade-offs

Understanding how metrics relate to each other is essential for interpreting the experimental results.

**Context relevance → faithfulness causality:**  
Context relevance is causally upstream of faithfulness. A low context relevance score means the retriever returned irrelevant passages. If the model is faithful to those irrelevant passages, faithfulness will be high but correctness will be low (the model faithfully repeated irrelevant information). If the model ignores the irrelevant context and draws on parametric knowledge, faithfulness will be low and risk will be high. Neither outcome produces a useful answer.

```
High CR + High Faith = Good retrieval AND good grounding → LOW RISK
Low CR  + High Faith = Poor retrieval, model faithful to irrelevant content → misleading LOW RISK
High CR + Low Faith  = Good retrieval, model ignores it  → HIGH RISK
Low CR  + Low Faith  = Total failure on both fronts      → HIGH RISK
```

**Latency vs. quality:**  
Hybrid retrieval adds BM25 scoring and RRF fusion to the pipeline, increasing retrieval latency. The key research question (RQ3) is whether this additional cost produces measurable improvements in context relevance and faithfulness. If the Pearson r between `e2e_ms` and `context_relevance` is positive and moderate-to-strong, it suggests the hybrid overhead is worthwhile.

**Token count vs. faithfulness:**  
Higher `top_k` or larger chunk sizes increase prompt token counts, which increases LLM processing time. Larger context windows may improve faithfulness by providing more supporting evidence, but may also increase the model's tendency to misrepresent long contexts (positional bias — Huang et al., 2023). The relationship between `total_tokens` and `answer_faithfulness` captures this empirically.

---

## Implementation Reference

### RAGASRunner

```python
# evaluation/ragas_runner.py

runner = get_ragas_runner()   # singleton

scores = runner.evaluate(
    query="What caused the 2008 financial crisis?",
    answer="The crisis was caused by...",
    contexts=["The 2008 financial crisis was triggered by..."],
    ground_truth="Subprime mortgage collapse",  # optional
)

# Returns:
# {
#   "context_relevance":   0.78,   # float in [0,1] or None
#   "answer_faithfulness": 0.84,   # float in [0,1] or None
#   "answer_correctness":  0.71,   # float in [0,1] or None
# }
```

**Error handling:** If RAGAS throws any exception (e.g., the judge model times out, or the API returns a malformed response), `evaluate()` catches the exception, logs a warning, and returns `None` for all scores. The pipeline continues — a single evaluation failure does not abort the experiment.

### HallucinationScore

```python
# evaluation/hallucination_score.py

from evaluation.hallucination_score import (
    compute_hallucination_risk,
    risk_band,
    risk_colour,
)

risk = compute_hallucination_risk(faithfulness=0.75, context_relevance=0.65)
# → 0.29 (LOW)

band  = risk_band(risk)    # → "LOW"
color = risk_colour(risk)  # → "#2ecc71"

# None handling
risk_none = compute_hallucination_risk(None, 0.8)
# → None

risk_band(None)    # → "UNKNOWN"
risk_colour(None)  # → "#95a5a6"
```

### Metrics Module

```python
# evaluation/metrics.py

from evaluation.metrics import descriptive_stats, cohens_d, pearson_r

# Descriptive statistics
values = [0.72, 0.81, 0.68, 0.79, 0.85]
stats = descriptive_stats(values, metric="answer_faithfulness")
print(stats.mean)    # 0.77
print(stats.iqr)     # Q3 - Q1
print(stats.to_dict())

# Cohen's d
d = cohens_d(
    group_a=[0.8, 0.75, 0.82],
    group_b=[0.6, 0.65, 0.58],
)
print(d.d)           # positive value → group_a higher
print(d.magnitude)   # "large"

# Pearson r
r = pearson_r(
    x=[100, 200, 150, 300],   # e2e_ms
    y=[0.7,  0.6,  0.8, 0.5], # faithfulness
)
# → negative r if higher latency correlates with lower faithfulness
```

### TokenCounter

```python
# telemetry/token_counter.py

from telemetry.token_counter import TokenCounter

counter = TokenCounter()

# Count tokens in any string
n = counter.count_tokens("What is Retrieval-Augmented Generation?")
# → e.g., 8

# Build usage from Ollama response (preferred — uses Ollama's own counts)
usage = counter.from_generation_response(gen_response)
print(usage.prompt_tokens)       # 847
print(usage.completion_tokens)   # 312
print(usage.estimated_cost_usd)  # 0.000000 (local inference)
```

---

## Worked Example

A single query through the RAGScope pipeline produces the following metric values:

```
Query:    "Which gene mutation is most commonly associated with hereditary
           haemochromatosis type 1?"
Dataset:  BioASQ (Phase A)
Model:    Llama 3 (8B)
Strategy: Hybrid retrieval

─── Retrieved Chunks (top 5, hybrid) ──────────────────────
  [1] score=0.043 | "HFE-related haemochromatosis is caused by mutations..."
  [2] score=0.041 | "The C282Y mutation in the HFE gene accounts for..."
  [3] score=0.038 | "TFR2 mutations cause a rarer, non-HFE form..."   ← different gene
  [4] score=0.035 | "Homozygosity for C282Y is found in the majority..."
  [5] score=0.031 | "Juvenile haemochromatosis involves HJV or HAMP..."  ← wrong subtype

─── Generated Answer ────────────────────────────────────────
  "Hereditary haemochromatosis type 1 is most commonly caused by the
   C282Y mutation in the HFE gene, usually in the homozygous state."

─── Telemetry ───────────────────────────────────────────────
  embed_query_ms :    38.2 ms
  bm25_search_ms :    12.1 ms
  rrf_fusion_ms  :     2.3 ms
  retrieval_ms   :    52.6 ms
  generation_ms  : 4,210.0 ms
  evaluation_ms  :18,340.0 ms   ← 3 judge-LLM calls (context relevance, faithfulness, correctness)
  e2e_ms         :22,602.6 ms   ← retrieval + generation + evaluation
  prompt_tokens  :      923
  completion_tokens:     47
  total_tokens   :      970
  estimated_cost :   $0.000000

─── RAGAS Evaluation ────────────────────────────────────────
  context_relevance   : 0.62   ← moderate; chunks [3] and [5] are off-topic noise
  answer_faithfulness : 0.91   ← high; answer grounded in chunks [2] and [4]
  answer_correctness  : 0.74   ← correct gene/mutation, phrasing differs from ground truth
  hallucination_risk  : 0.16   ← LOW: 1 − (0.6×0.91 + 0.4×0.62) = 0.16
```

**Reading this result:** The retrieval returned some irrelevant passages (chunks [1], [3], [5] relate to wrong birth years or different people), reducing context relevance to 0.62. However, the generative model correctly identified the relevant passages [2] and [4] and produced a faithful answer, resulting in high faithfulness (0.91). The composite hallucination risk is LOW at 0.16. The overall answer quality is good (correctness 0.74), with the small shortfall attributable to the query's birth-year constraint not being fully resolved by the retriever.

---

## Known Limitations

**Metric sensitivity to judge model quality:** All three RAGAS metrics depend on the judge LLM's ability to accurately decompose claims, identify relevance, and assess semantic similarity. A weaker judge model will produce less reliable metric values. The dissertation uses Llama 3 (8B) as the judge, which is capable but not state-of-the-art for meta-reasoning tasks.

**Faithfulness for short answers:** The atomic claim decomposition step of faithfulness scoring assumes multi-sentence answers. For one-word answers ("yes", "Paris", "1989"), the claim decomposition is trivial or undefined, and faithfulness scores for these cases may not be meaningful.

**Context relevance granularity:** RAGAS context relevance operates at the sentence level, which may be too coarse for very short chunks (< 3 sentences). Chunks smaller than 3 sentences may receive binary rather than fractional scores.

**Hallucination risk is a proxy, not ground truth:** The hallucination risk score is a proxy derived from two imperfect metrics, not a direct measurement of factual incorrectness. A HIGH risk score means the answer is likely unfaithful to the retrieved context — it does not mean every claim is necessarily wrong (the model may be drawing on correct parametric knowledge). Correctness verification requires a human annotator or a separate fact-checking system.

**Pearson r assumes linearity:** The latency–quality correlations reported in the analysis assume a linear relationship. If the true relationship is non-linear (e.g., quality improves up to a latency threshold then plateaus), Pearson r will underestimate the strength of the relationship.

---

## References

- Barnett, S. et al. (2024) 'Seven failure points when engineering a retrieval augmented generation system', *ICAIE 2024*
- Cohen, J. (1988) *Statistical Power Analysis for the Behavioural Sciences*, 2nd edn. Hillsdale, NJ: Lawrence Erlbaum Associates
- Es, S. et al. (2023) 'RAGAS: Automated evaluation of retrieval augmented generation', *arXiv:2309.15217*
- Huang, L. et al. (2023) 'A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions', *arXiv:2311.05232*
- Ji, Z. et al. (2023) 'Survey of hallucination in natural language generation', *ACM Computing Surveys*, 55(12), pp. 1–38