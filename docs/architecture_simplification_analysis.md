# RAGScope — Architecture Simplification Analysis

> Prepared in response to supervisor feedback: the platform is sound, but the architecture
> needs simplifying — removing what isn't load-bearing for the dissertation and reducing
> abstraction so the system is easier to explain in a viva or a diagram.
>
> **This is a read-only analysis. No code has been changed.** Every finding below is
> verified against the actual current codebase (file/line references given throughout),
> not assumed from documentation. Where I recommend removal, I also state the blast radius
> (what depends on it, what breaks) so the decision is informed, not just directional.

---

## How to read this document

Findings are grouped by *where* the complexity lives, roughly in order of how confidently
"remove this" vs. how much it's a judgement call:

1. [Dependency-level bloat](#1-dependency-level-bloat) — highest confidence, zero functional risk
2. [Docker/deployment-level bloat](#2-dockerdeployment-level-bloat) — high confidence
3. [Dead code](#3-dead-code) — high confidence, verified by exhaustive grep
4. [Diagram-level complexity](#4-diagram-level-complexity-the-real-supervisor-ask) — the actual point of the feedback
5. [Judgement calls](#5-judgement-calls--optional-simplifications) — real trade-offs, your call
6. [What NOT to touch](#6-what-not-to-touch--necessary-complexity) — components that look complex but are load-bearing for the dissertation's own claims

---

## 1. Dependency-level bloat

**Finding: 5 of ~24 declared runtime dependencies in `pyproject.toml` are never imported anywhere in the codebase.**

Verified by exhaustive grep across every `.py` file (excluding `.venv`):

| Package | Declared as | Actually used? |
|---|---|---|
| `ir-datasets` | "MS MARCO loader" | **No.** `data/loaders/msmarco.py` uses HuggingFace `datasets.load_dataset("microsoft/ms_marco", ...)`, not `ir_datasets`, anywhere. Zero imports. |
| `nltk` | (uncommented) | **No.** Zero imports anywhere. |
| `altair` | "Declarative visualisation" | **No.** Every chart in `dashboard/components/` (`heatmap.py`, `latency_chart.py`) uses Plotly exclusively. Zero imports. |
| `scipy` | "Statistical analysis (Cohen's d)" | **No — and this one actively contradicts your own design principle.** `docs/architecture.md` states as a deliberate design choice: *"Pure-Python statistics. `evaluation/metrics.py` has no numpy or scipy dependency... This makes the module testable without the full ML stack and documents the statistical formulae explicitly."* `evaluation/metrics.py` genuinely does implement Cohen's d and Pearson r from first principles (confirmed by reading the file). `scipy` is dead weight that contradicts the very design principle you'd want to point to in a viva. |
| `rich` | "Terminal output formatting" | **No.** Zero imports anywhere. |

**Why this matters for "explainability":** a supervisor (or you, defending the dissertation) reading `pyproject.toml` to understand what the system depends on will draw false conclusions — e.g. "why does this need NLTK if the RAGAS/tiktoken pipeline handles NLP?" There's no good answer, because it doesn't. Every unused dependency is a question you'd have to deflect rather than answer.

**Recommendation:** remove all 5 from `pyproject.toml`. Zero functional risk — nothing imports them, so nothing breaks. This alone cuts the declared dependency surface by ~20%.

---

**Finding: the `[project.scripts]` console-script entry points are entirely unused, and one is broken.**

```toml
[project.scripts]
ragscope-ingest    = "pipeline.ingestion:main"
ragscope-query     = "pipeline.rag:main"
ragscope-benchmark = "experiments.run_2x2_factorial:main"
ragscope-analyse   = "experiments.analyse_results:main"
ragscope-dashboard = "dashboard.app:main"
```

Verified:
- Grepped every `.md`, `.sh`, and `Dockerfile*` in the repo for any of these five command names — **zero references anywhere.** Every documented invocation throughout the entire project (README, `docs/architecture.md`, `docs/datasets.md`, the Dockerfile, `entrypoint.sh`) uses `uv run python <module_path>.py` directly, never the console-script form.
- `ragscope-dashboard = "dashboard.app:main"` is **not just unused, it's broken**: `dashboard/app.py` has no `main()` function at all (confirmed — it's a plain top-to-bottom Streamlit script, as all `dashboard/pages/*.py` files are). Even if it did, a Streamlit app can't be correctly launched by a plain Python function call from a console-script shim — it needs the `streamlit run` bootstrap to set up the server/session context. This entry point would fail immediately if anyone ran `ragscope-dashboard`.

**Recommendation:** remove the entire `[project.scripts]` section. It documents a way of running the system that nobody uses and that doesn't fully work. Keeping it invites a supervisor or examiner to try `ragscope-dashboard` (a plausible thing to try) and hit an error.

---

## 2. Docker/deployment-level bloat

**Finding: the production image unconditionally bundles Jupyter/notebook dependencies, and does so redundantly (twice).**

`docker/app/Dockerfile`, builder stage:

```dockerfile
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --all-extras --frozen --no-install-project --no-dev
    #        ^^^^^^^^^^^^ already installs the "notebooks" extra here

COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ── optional notebooks (only if needed) ───────────
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra notebooks
    #                          ^^^^^^^^^^^^^^^^^ redundant — --all-extras above already did this
```

Two separate problems here:

1. **`--all-extras` on the first sync already installs `jupyterlab`, `ipywidgets`, `matplotlib`, and `seaborn`** into the image that serves the production dashboard (`ragscope` service) — a service that never uses any of them. This inflates the shipped research artefact's image size and dependency surface with packages that exist solely to support exploratory notebook analysis, which is a separate, dev-only concern (see `docker-compose.override.yml`'s `jupyter` service).
2. **The later "optional notebooks" `RUN` step is a complete no-op** — it re-syncs the same extra that `--all-extras` already pulled in, adding a wasted Docker layer and build time with a comment ("only if needed") that isn't actually gating anything.

**Why this matters:** the core claim of the dissertation's architecture is a lean, purpose-built observability platform. Shipping `jupyterlab` inside the production `ragscope:latest` image undercuts that story and adds real build weight for zero functional benefit to the platform being evaluated.

**Recommendation:** change the first sync to `uv sync --frozen --no-install-project --no-dev` (drop `--all-extras`), and delete the redundant "optional notebooks" `RUN` block entirely. If you still want the `jupyter` dev service to work, either (a) give it its own minimal Dockerfile that installs the `notebooks` extra, or (b) accept that the dev-only override service installs the extra itself at container start — either way, the *production* image stops carrying dead weight.

---

**Finding: the Jupyter service and `notebooks/` directory are not part of the research platform's architecture — they're a separate exploratory-analysis convenience.**

`docs/images/ragscope_system_architecture.svg` currently draws `jupyter` as a first-class box in the "Deployment topology" row, on equal visual footing with `chromadb` and `ragscope`. But:
- Nothing in Chapter 3's system description (§3.6.1) treats Jupyter as part of the platform — it isn't a RAG pipeline component, a telemetry component, or a dashboard component. It's a notebook environment for you to poke at data during development.
- It's explicitly dev-only (`docker-compose.override.yml`, not the base `docker-compose.yml`), consistent with it not being part of what gets evaluated.

**Why this matters:** a system architecture diagram meant to be explained in a viva should draw a hard line between "the artefact I built and evaluated" and "tools I used while building it." Jupyter is the latter. Drawing it as an equal-weight box invites the question "what does this contribute to the RAG pipeline?" — to which the honest answer is "nothing, it's for me."

**Recommendation:** drop the `jupyter` box from the core architecture diagram entirely (see [Section 4](#4-diagram-level-complexity-the-real-supervisor-ask)). This is a diagram-level change, not a code change — the notebooks and the dev override file can stay in the repo untouched; they just shouldn't appear in the diagram you present as "the architecture."

---

## 3. Dead code

Every item below was verified by grepping the **entire codebase** (application code, dashboard, experiments, and tests) for callers — not just "looks unused," but "zero call sites found anywhere, including in tests."

| Location | Dead symbol | Evidence |
|---|---|---|
| `pipeline/generation/llm_client.py:163` | `BaseLLMClient.is_available()` | Zero callers anywhere (not `pipeline/rag.py`, not dashboard, not tests). |
| `pipeline/generation/llm_client.py:172` | `BaseLLMClient.list_available_models()` | Zero callers anywhere. |
| `pipeline/generation/llm_client.py:225` | `BaseLLMClient.close()` | Zero callers anywhere — the underlying `httpx.Client` is never explicitly closed; it's just garbage-collected. |
| `pipeline/vectorstore.py:297` | `VectorStore.delete_chunks()` | Zero callers anywhere, including tests. This is a leftover from the original Knowledge Base page's "add/remove chunk" forms, which were fake (no working handlers) and have since been replaced with the current read-only browser. There is nothing left in the codebase that deletes individual chunks. |
| `dashboard/components/latency_chart.py:131` | `latency_histogram()` | Zero callers anywhere. None of the 5 dashboard pages import it; only `latency_time_series`, `latency_box_by_condition`, and `pipeline_flow_sankey` from this module are actually used. |
| `telemetry/token_counter.py:64,68` | `TokenCounter.count_tokens()`, `count_prompt()` | Only exercised by `tests/test_telemetry.py` — **never called by the live pipeline.** `pipeline/rag.py` only ever calls `TokenCounter.from_generation_response()`, because (per Chapter 3 §3.6.5) actual token counts come directly from Ollama's response metadata, not from re-tokenising the prompt. These two methods appear to be exactly the tool used for the one-off validation exercise Chapter 3 describes ("empirically verified to be consistent with independent tiktoken counts on a validation sample of 50 queries") rather than something the running system depends on. |

**Recommendation:**
- `is_available`, `list_available_models`, `close`, `delete_chunks`, `latency_histogram`: straightforward removal, no user-facing or research-facing impact, no test breakage (none are tested except by their own dead existence).
- `count_tokens`/`count_prompt`: lower priority — these have a legitimate one-time purpose (the tiktoken cross-validation you already did and reported in Chapter 3), they're just not part of the live pipeline. Options: (a) leave them, since they're cheap and documented, or (b) move them out of the production `telemetry/token_counter.py` module into a small one-off validation script under a `scripts/` or similar location, so the production module only contains what the live pipeline actually uses. Your call — this is more "tidiness" than "bloat."

---

## 4. Diagram-level complexity (the real supervisor ask)

This is very likely what "simplify the architecture" actually means in practice — not primarily the Python code (which is, module-for-module, reasonably lean — see [Section 6](#6-what-not-to-touch--necessary-complexity)), but the **system architecture diagram** trying to show everything at once.

I extracted every text label currently drawn in `docs/images/ragscope_system_architecture.svg`. It draws **~37 distinct boxes across 7 layers**:

| Layer | Boxes drawn | Count |
|---|---|:---:|
| Deployment topology | `chromadb`, `ragscope`, `ollama`, `jupyter`, `config/settings.py` | 5 |
| Data ingestion pipeline | MS MARCO, Natural Questions, HotpotQA, `cleaner.py`, `chunker.py`, `embeddings.py`, `ingestion.py`, ChromaDB, `bm25_corpus.jsonl`, `ingestion_manifest` | 10 |
| RAG query pipeline | User query, `dense.py`, `hybrid.py`, `llama3.py`, `mistral.py`, `llm_client.py` | 6 |
| Telemetry layer | `timer.py`, `token_counter.py`, `logger.py`, `telemetry/store` | 4 |
| Evaluation layer | `ragas_runner.py`, `hallucination_score.py`, `metrics.py`, output-fields box | 4 |
| Streamlit dashboard | 5 individual pages + shared components box | 6 |
| Experiment runner | `run_2x2_factorial.py`, `analyse_results.py` | 2 |

**37 boxes is too many for a diagram meant to be grasped at a glance and defended verbally.** Every one of these boxes maps to something real (nothing here is invented), but a diagram's job is to communicate the *shape* of the system, not to be a 1:1 file listing. Right now it's closer to a file listing with colour-coding than an architecture diagram.

Concretely, several groups of boxes are siblings at the wrong level of granularity for a top-level diagram:
- The three benchmark datasets (MS MARCO / NQ / HotpotQA) are conceptually **one input**: "three benchmark corpora." Their individual sampling strategies matter enormously for Chapter 3's *methodology* text, but not for the *architecture* diagram — a reader doesn't need three boxes to understand "data goes in."
- `cleaner.py` → `chunker.py` → `embeddings.py` → `ingestion.py` is a single **preprocessing pipeline**; showing four sequential boxes for what is conceptually one "clean → chunk → embed → index" stage adds arrows and nodes without adding understanding.
- `llama3.py` / `mistral.py` / `llm_client.py` is one **generation stage** with two swappable model backends — exactly parallel to how `dense.py`/`hybrid.py` is one **retrieval stage** with two swappable strategies. Right now retrieval gets 2 boxes for 2 strategies and generation gets 3 boxes for 2 models plus their shared base — inconsistent granularity for what's conceptually the same kind of thing (a pluggable strategy pair).
- `timer.py` + `token_counter.py` are both just **instrumentation feeding `logger.py`** — three boxes for what is, from an architecture point of view, one "measure everything, write it as JSON" stage.
- The dashboard's 5 pages are individually useful to know about (and Chapter 1, Objective 5 specifically claims "five dashboard views" as evidence), but they don't need to be 5 separate boxes in the *architecture* diagram — a single "Dashboard (5 views)" box with the 5 names listed as a caption/sub-label preserves the evidentiary detail without spending 5 node-slots on it.
- The "Deployment topology" row (Docker services, ports) is a genuinely different *kind* of diagram from the rest (infrastructure/ops vs. research data-flow) and mixing them is part of why the current diagram feels heavy. A research architecture diagram's job is to show data flowing through pipeline stages; a deployment diagram's job is to show which process talks to which over which port. Conflating them means every viewer has to mentally filter out the one they don't currently care about.

I'll turn this into a concrete proposed structure in the companion plan document (`docs/architecture_simplification_plan.md`) rather than duplicate it here — this section is the diagnosis, that document is the proposal for you to react to.

---

## 5. Judgement calls — optional simplifications

These are real trade-offs, not clear-cut removals. I'm flagging them rather than recommending a direction.

**`_reconnect_on_stale_collection` decorator in `pipeline/vectorstore.py` (applies to 7 methods).** This exists to fix a real bug you hit during development: if the ChromaDB collection is deleted and recreated by one process (e.g. `ingestion.py --reset`) while another process (e.g. the dashboard) holds a cached handle to the old collection UUID, every call on the stale handle raises `NotFoundError`. The decorator catches this and reconnects once, transparently.
- *In favour of keeping:* it fixes a bug that has actually occurred in this project, not a hypothetical one.
- *In favour of simplifying:* if your actual usage pattern is "run ingestion, then separately open the dashboard" (never both processes touching the same collection concurrently), the decorator is defending against a scenario that may not occur in your real workflow, and it does add a layer of indirection (`functools.wraps`, exception-catching wrapper) around 7 methods that a reader has to understand once to follow any of them.
- If you simplify: the risk is that if you *do* ever run `ingestion.py --reset` while the dashboard is open in another terminal, you'd see a crash again and need to restart the dashboard process — annoying but not data-destructive.

**Docker Compose override file mechanism** (`docker-compose.yml` + `docker-compose.override.yml`, merged via `-f` flags or `COMPOSE_FILE`). This is a standard, well-understood Docker Compose pattern, but it is a second file a reader needs to know exists and how it merges with the first. Given the override file's only job is "add a dev-only Jupyter service and a couple of environment/volume tweaks," you could fold the Jupyter service into the main `docker-compose.yml` behind a Compose *profile* (`profiles: ["dev"]`) instead of a second file — one file, one mental model, same opt-in behaviour (`docker compose --profile dev up`). Not obviously simpler or more complex than the current approach; a style choice.

**Five dashboard pages vs. fewer, denser pages.** Not evaluated in depth here since this is a UX/scope question rather than a backend architecture one, but worth naming: Live Monitor, Comparison, Query Explorer, Benchmark Results, and Knowledge Base each have a clear, distinct purpose tied to your research questions (RQ1–RQ3) and are independently referenced as evidence for the SMART objectives in Chapter 1. Consolidating them would reduce box count in a dashboard-specific diagram but would blur which page evidences which objective — probably not worth doing purely for diagram simplicity when the "show 1 box, list 5 names" treatment already solves the visual complexity problem without touching the code.

---

## 6. What NOT to touch — necessary complexity

Equally important for your review: several things that *look* like complexity when read as code, but are directly defending specific, load-bearing claims in the dissertation. Removing or simplifying these would weaken the methodology, not just the diagram.

- **`pipeline/retrieval/dense.py` + `hybrid.py` as two separate classes.** This is exactly RQ3's independent variable (retrieval strategy). They share almost no duplicable logic beyond "embed the query" — dense is a single ANN search, hybrid is ANN + BM25 + RRF fusion. Two genuinely different algorithms, correctly separated.
- **`pipeline/generation/llm_client.py` + `llama3.py` + `mistral.py` (ABC + 2 subclasses).** Looks like inheritance overhead until you notice *why* it exists: Llama 3 and Mistral use different, incompatible chat templates (`<|begin_of_text|>...<|eot_id|>` vs `<s>[INST]...[/INST]`), and getting the template wrong measurably degrades faithfulness. The base class holds everything template-agnostic (HTTP calls, retry, token extraction); the two subclasses hold only the ~10-line `_build_prompt` override each. This is about as small as this abstraction can get while still being correct — collapsing it into one class with an `if model == "llama3"` branch would be marginally shorter but not actually simpler to reason about, and would make it harder to point at "here is exactly what differs between conditions."
- **`evaluation/ragas_runner.py`'s "rebuild the judge LLM fresh every call" pattern.** This looks like unnecessary object churn until you read the docstring: it's the fix for a genuine, previously-encountered bug (`ragas.evaluate()` opens a new asyncio event loop per call; a cached `ChatOllama` client's connections don't survive across loops, silently producing NaN scores on the second call onward). This is necessary complexity directly caused by an external library's design, not something RAGScope chose to add.
- **`evaluation/metrics.py` being pure Python (no numpy/scipy).** Already a deliberate, documented design principle in `docs/architecture.md`, and one worth keeping *and defending* in a viva: "I implemented Cohen's d and Pearson r from the textbook formulae myself so the computation is fully auditable, not delegated to a library black box." This is a strength, not complexity to cut.
- **Per-stage `Timer` instrumentation (embed / retrieval / generation / evaluation) feeding a flat telemetry record.** This is literally what RQ2 (fault localisation) and Objective 3 (telemetry layer) claim as a contribution — the platform's ability to localise latency to a specific stage is the point, not incidental detail.

---

## Summary table

| # | Finding | Confidence | Effort to fix | Risk if removed |
|---|---|---|---|---|
| 1 | 5 unused dependencies in `pyproject.toml` | Very high | Trivial | None |
| 2 | Broken/unused `[project.scripts]` entry points | Very high | Trivial | None |
| 3 | Notebooks extra baked unconditionally + redundantly into prod image | High | Low | None (Jupyter dev service needs its own install path) |
| 4 | Jupyter box in architecture diagram | High | Diagram-only | None |
| 5 | 5 dead methods/functions across 3 files | High | Trivial | None |
| 6 | `count_tokens`/`count_prompt` unused in live pipeline | Medium | Low | None (documented one-off validation use) |
| 7 | Diagram has ~37 boxes across 7 mixed-granularity layers | High (diagnosis) | See plan doc | N/A — diagram change only |
| 8 | `_reconnect_on_stale_collection` decorator | Judgement call | Medium | Re-introduces a real, previously-hit bug if removed |
| 9 | Compose override file vs. profile | Judgement call | Low | None either way |

**Nothing in this document has been applied to the code.** Once you've reviewed this and the companion plan document, tell me which items to act on and I'll implement only those, in the order you prefer.
