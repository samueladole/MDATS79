# RAGScope — Architecture Simplification Changelog

> Record of what was actually applied, following supervisor feedback to simplify the
> architecture. See `docs/architecture_simplification_analysis.md` for the original
> diagnosis and rationale, and `docs/architecture_simplification_plan.md` for the
> diagram redesign proposal both were reviewed and approved against before any code
> was touched. This document is the "what actually happened" record; the other two
> are the "what was proposed and why."

**Dates:** 2026-07-26 – 2026-07-27
**Scope:** Documentation, architecture diagram, dependency manifest, Docker build, and dead-code removal. No changes to pipeline/evaluation/telemetry *logic* — RAGAS metrics, hallucination scoring, retrieval strategies, and the 2×2 factorial design are all untouched.

---

## 1. Architecture diagram

- **`docs/images/ragscope_system_architecture.svg`** — redrawn from scratch. ~37 boxes across 7 mixed-granularity layers (research pipeline interleaved with Docker deployment topology) → **15 boxes across 6 stages**, each stage explicitly labelled with the research question or SMART objective it evidences (Data → RAG Query Pipeline → Evaluation → Telemetry → Dashboard → Experiments).
- Deployment/Docker topology dropped from the diagram entirely — it's a different kind of diagram (infrastructure vs. research data-flow) and is already covered in prose in this file's [Service Layer (Docker)](architecture.md#service-layer-docker) section and Chapter 3 §3.6.1.
- `metrics.py` ("Statistics") drawn once as a standalone cross-cutting node with arrows into both the Dashboard and Experiments stages, reflecting how it's actually used (not nested inside the per-query Evaluation stage, which never calls it).
- Colour legend reduced from 7 categories to 6 (Teal/Purple/Coral/Amber/Blue/Green) — no separate "Gray/shared" category, since the Statistics node now lives inside the Coral (evaluation) family.
- Every box's rendered text width was verified against its container width via font-metric measurement (PIL) before finalising coordinates, then the whole diagram was rendered through headless Chrome to visually confirm layout, following the same verification method used earlier in the project to catch a text-overflow bug in the previous diagram revision.
- **`docs/images/archive/ragscope_system_architecture_v1_full.svg`** — the pre-simplification diagram, archived unchanged for reference.

---

## 2. Dependency manifest (`pyproject.toml`)

Removed 5 declared runtime dependencies with zero imports anywhere in the codebase (verified by exhaustive grep):

| Package | Was declared as | Actual usage found |
|---|---|---|
| `ir-datasets` | "MS MARCO loader" | None — `data/loaders/msmarco.py` uses `datasets.load_dataset("microsoft/ms_marco", ...)` |
| `nltk` | (uncommented) | None |
| `altair` | "Declarative visualisation" | None — dashboard uses Plotly exclusively |
| `scipy` | "Statistical analysis (Cohen's d)" | None — `evaluation/metrics.py` is deliberately pure Python (see Design Principles in `docs/architecture.md`); this entry directly contradicted that documented design choice |
| `rich` | "Terminal output formatting" | None |

Also removed the entire `[project.scripts]` section — five console-script entry points (`ragscope-ingest`, `ragscope-query`, `ragscope-benchmark`, `ragscope-analyse`, `ragscope-dashboard`), none referenced anywhere in documentation or scripts (every documented command uses `uv run python <module>.py` directly), and `ragscope-dashboard = "dashboard.app:main"` was non-functional — `dashboard/app.py` has no `main()` function, and a Streamlit app can't be correctly launched via a plain console-script shim regardless.

**Verified impact:** `uv lock` + `uv sync` removed **15 packages** from the environment — `ir-datasets` and `nltk` alone were dragging in 13 transitive dependencies (`lxml`, `beautifulsoup4`, `trec-car-tools`, `warc3-wet`, etc.) with no connection to this project. `altair`, `scipy`, and `rich` remain in the resolved lock as legitimate transitive dependencies of `streamlit`, `statsmodels`, and `typer` respectively — expected and correct; the fix was that `pyproject.toml` no longer *directly* claims this project needs them.

---

## 3. Docker build (`docker/app/Dockerfile`, `docker-compose.yml`, `docker-compose.override.yml`)

- Dropped `--all-extras` from the dependency-only build layer in `docker/app/Dockerfile`. Previously this unconditionally installed `jupyterlab`/`matplotlib`/`seaborn`/`ipywidgets` into the **production** `ragscope` image, even though only the dev-only `jupyter` override service uses them.
- Deleted the subsequent "optional notebooks" `RUN uv sync --extra notebooks` layer — it was a complete no-op, since `--all-extras` on the earlier layer had already installed the same packages.
- **First attempt (superseded):** had the `jupyter` service `pip install jupyterlab ipywidgets matplotlib seaborn` as the first step of its container command, before launching `jupyter lab`. This worked but re-ran the full install — a ~100-package resolve and download — on every single container start, since nothing cached it. Reported as a real regression in practice.
- **Fix:** added a dedicated build stage instead. `docker/app/Dockerfile` now has `builder-notebooks` (extends `builder`, syncs the `notebooks` extra) and a `jupyter` runtime stage (copies that venv, mirrors the `app` stage otherwise). `jupyterlab`/`matplotlib`/`seaborn`/`ipywidgets` are now baked in at **build** time, cached by Docker like any other layer — `docker compose up` no longer re-downloads anything, only `docker compose build jupyter` does, and only when `pyproject.toml`/`uv.lock` change. The `jupyter` service now builds this stage via `build.target: jupyter` and tags the result `ragscope-jupyter:latest` (a distinct image from `ragscope:latest`, since its contents genuinely differ).
- **Bug found and fixed while making the above change:** adding a stage *after* `app` in the Dockerfile meant Docker's "build the last stage when none is specified" default silently changed what `docker-compose.yml`'s `ragscope` service built — `target:` had never been set on it because there was only ever one runtime stage before. Verified this was a real, live bug (not theoretical): building `ragscope` produced an image with `jupyterlab` importable inside it. Fixed by adding `target: app` explicitly to the `ragscope` service.
- **Verified:** built both targets from a clean Docker cache — `ragscope:latest` (442 MB) confirmed to have neither `jupyterlab` nor `matplotlib` importable; `ragscope-jupyter:latest` (491 MB) confirmed to have all four notebook packages importable. Rebuilding `jupyter` a second time hit 100% cache with zero downloads. `docker compose -f docker-compose.yml -f docker-compose.override.yml config` resolves cleanly with both services' explicit targets intact.

---

## 4. Dead code removal

All confirmed by exhaustive grep across application code, dashboard, experiments, and tests — zero call sites anywhere before removal.

| File | Removed | Notes |
|---|---|---|
| `pipeline/generation/llm_client.py` | `BaseLLMClient.is_available()`, `.list_available_models()`, `.close()` | No callers anywhere |
| `pipeline/vectorstore.py` | `VectorStore.delete_chunks()` | Leftover from the Knowledge Base page's original fake add/remove forms; the page has been read-only since that rebuild |
| `dashboard/components/latency_chart.py` | `latency_histogram()` | No dashboard page imports it (only `latency_time_series`, `latency_box_by_condition`, `pipeline_flow_sankey` are used) |
| `telemetry/token_counter.py` | `TokenCounter.count_tokens()`, `.count_prompt()` | Only exercised by tests, never by the live pipeline (`RAGPipeline.query()` only calls `.from_generation_response()`, since Ollama reports token counts directly). These existed to support the one-off tiktoken cross-validation described in Chapter 3 §3.6.5 |

Corresponding dead tests removed from `tests/test_telemetry.py` (`test_count_tokens_nonempty`, `test_count_tokens_empty`, `test_count_prompt_includes_context`) — no other test files referenced any of the removed symbols.

---

## 5. Pre-existing test failures fixed (found during verification, unrelated to the cleanup)

Two tests were failing *before* any of the above changes — confirmed by stashing the cleanup and re-running against the untouched codebase. Root-caused and fixed rather than left as noise:

- **`tests/test_evaluation.py::TestCohensD::test_large_effect`** — fixture used identical repeated values within each group (`[0.9]*5` vs `[0.1]*5`), giving zero within-group variance. `cohens_d()` correctly guards against a zero pooled standard deviation by returning `d=0.0` rather than dividing by zero, so the test could never assert "large" regardless of the gap between groups. Fixed by using realistic clustered-but-non-identical values.
- **`tests/test_telemetry.py::TestTelemetryLogger::test_log_and_load`** — asserted a nested `record["token_usage"]["prompt_tokens"]` shape, but `build_record()` deliberately uses a flat schema (documented in `telemetry/logger.py`: *"intentionally flat to simplify pandas ingestion"*), with token fields at the top level. Fixed the assertions to match the actual, correct schema.

**Test suite: 66/66 passing** (was 67/69 passing pre-cleanup, including the 2 unrelated failures above; 3 tests for the removed `count_tokens`/`count_prompt` methods were deleted along with the methods they tested).

---

## 6. Judgment calls — resolved, no code changed

Two items were flagged in the analysis document as genuine trade-offs rather than clear-cut removals. Both were resolved by asking directly rather than guessing:

- **`_reconnect_on_stale_collection` decorator** (`pipeline/vectorstore.py`, wraps 7 methods) — **kept**. It fixes a real crash (stale ChromaDB collection handle after a concurrent `--reset`) that does occur in the actual workflow: the dashboard is sometimes left running while `ingestion.py --reset` is run separately. Removing it would reintroduce that crash.
- **Compose override file vs. Compose profile** — **kept as an override file** (`docker-compose.override.yml`), no migration to a `profiles:`-gated service in the single compose file. Explicit preference, not a defect either way.

---

## 7. Documentation reconciliation (`docs/architecture.md`)

Rechecked in full against everything above and corrected:

- Removed `.delete_chunks(chunk_ids)` from the `pipeline/vectorstore.py` Component Reference table.
- Removed `.is_available()` from the `pipeline/generation/llm_client.py` Component Reference table.
- Removed `.count_tokens(text)` / `.count_prompt(query, contexts)` from the `telemetry/token_counter.py` Component Reference table, and added a note explaining why (dead code, superseded by `.from_generation_response()`, with a pointer back to the Chapter 3 validation exercise they originally supported).
- Added a paragraph to the Service Layer (Docker) section documenting that the production image no longer bundles notebook dependencies (later updated again — see below — once the build-time-stage fix in §3 landed).
- **Unrelated to this pass, found opportunistically while doing the full recheck:** both the "High-Level Architecture" ASCII diagram and the "Module Dependency Map" ASCII diagram were missing the Knowledge Base page (`05_knowledge_base.py`) — they'd gone stale when that page was added earlier in the project, predating this simplification effort. Fixed both while already in the file.

No other sections required changes at that time — the RAG pipeline description, telemetry record schema, RAGAS judge configuration, retrieval algorithms, experimental conditions, and configuration reference were all already accurate and unaffected by this pass.

**Follow-up update, once the §3 build-time-stage fix landed:** `docs/architecture.md`'s Service Layer section and `README.md`'s Docker Reference section were both updated again to describe the corrected approach (notebook dependencies baked in at build time via a dedicated `jupyter` target, not `pip install`-ed at container start), and to note that both `ragscope` and `jupyter` now pin an explicit Compose `target:`.
