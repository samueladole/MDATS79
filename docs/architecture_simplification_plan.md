# RAGScope — Architecture Simplification Plan

> Companion to `docs/architecture_simplification_analysis.md`. That document diagnoses
> *what* is more complex than it needs to be and *why*. This document proposes *what a
> simplified architecture would look like* — concretely, in terms of the boxes and layers
> you'd draw — so you can react to a proposal rather than start from a blank page.
>
> **Nothing has been changed yet.** This is a plan for you to mark up, cut, or redirect.
> Once you've reviewed both documents and told me what to keep/change, the next step is
> for me to redraw `docs/images/ragscope_system_architecture.svg` to match what you've approved,
> and only then touch any code (per your instruction).

---

## 1. The goal, restated

Your supervisor's feedback, as you relayed it: the platform itself is good — the ask is to
**reduce complexity and remove what isn't relevant to the dissertation**, so that the
architecture is easier to explain (in the diagram, and by extension in the viva).

That means this plan optimises for one thing: **can someone unfamiliar with the codebase
look at the diagram for 60 seconds and correctly explain, out loud, what the system does
and how the four research-relevant stages connect?** Not "does the diagram show every
file," which is closer to what the current version does.

## 2. Two separate diagrams, not one

The single biggest complexity reducer available to you costs nothing in code and doesn't
remove any capability: **stop drawing the research pipeline and the deployment topology
as one diagram.**

- **Diagram A — Research / System Architecture.** The thing you defend in the viva. Data
  in → retrieval → generation → evaluation → telemetry → dashboard/experiments out. This
  is what Chapter 3 §3.6 describes in prose; the diagram should be its visual mirror,
  nothing more.
- **Diagram B — Deployment Topology** (optional, appendix-level). Docker services, ports,
  host vs. container boundary. Useful for anyone trying to *run* the system, not useful for
  anyone trying to *understand the research contribution*. Keep it simple (2 boxes:
  `ragscope` app + `chromadb`, with a one-line note "Ollama runs on the host") or drop it
  from the dissertation entirely and let the README/architecture.md carry that detail in
  text form instead of a diagram.

Everything below assumes Diagram A is the one you present as "the architecture." If you
want to keep a deployment diagram too, it should be visually distinct (separate figure,
separate page) and much smaller than what's in the "Deployment topology" row today.

## 3. Proposed box count: ~37 → ~16

Below is a concrete proposal, stage by stage, mapping today's boxes to a smaller set of
tomorrow's boxes. Nothing here removes a *capability* — every current box's content is
still represented, just grouped at a level that matches what a reader needs from an
architecture diagram rather than a file listing.

### Stage 1 — Data (today: 10 boxes → proposed: 3 boxes)

| Today | Proposed |
|---|---|
| Corpus loading (3 boxes) | **1 box: "BioASQ corpus"** — the individual sampling strategy (60/40/100, role-mapping heuristic, etc.) belongs in Chapter 3 prose and Table 3.2-style detail, not as diagram nodes. |
| `cleaner.py`, `chunker.py`, `embeddings.py`, `ingestion.py` (4 boxes) | **1 box: "Preprocessing & Indexing"** with a sub-caption "clean → chunk (512/64) → embed (MiniLM-384d) → index." One arrow in, one arrow out. |
| ChromaDB, `bm25_corpus.jsonl`, `ingestion_manifest` (3 boxes) | **1 box: "Retrieval Index"** (vector store + BM25 file) — the manifest is a build artefact, not an architectural component; mention it in a caption if at all, not as a node. |

### Stage 2 — RAG Query Pipeline (today: 6 boxes → proposed: 3 boxes)

| Today | Proposed |
|---|---|
| `dense.py`, `hybrid.py` (2 boxes) | **Keep as 2 boxes** — this is the actual independent variable for RQ3 (retrieval strategy), it deserves to stay visible as a fork in the diagram. |
| `llama3.py`, `mistral.py`, `llm_client.py` (3 boxes) | **1 box: "LLM Generation (Ollama)"** with a caption "Llama 3 8B / Mistral 7B — model-specific chat templates." This is the other independent variable for RQ3; showing it as *one* box with two named options inside keeps it visually parallel to how retrieval is drawn as a fork, rather than getting 3 boxes for what's conceptually the same kind of thing (2 swappable strategies + shared plumbing). |
| "User query" (1 box) | **Keep** — it's the entry point, costs nothing to keep. |

### Stage 3 — Telemetry (today: 4 boxes → proposed: 2 boxes)

| Today | Proposed |
|---|---|
| `timer.py`, `token_counter.py`, `logger.py` (3 boxes) | **1 box: "Instrumentation → Telemetry Record"** — caption: "per-stage latency (embed/retrieve/generate/evaluate) · token counts · cost." This is one conceptual stage: measure everything, write it as one JSON record. |
| `telemetry/store` (1 box) | **Keep as the output** — it's the thing both the dashboard and the experiment runner read from, worth showing as a distinct node since two other stages depend on it. |

### Stage 4 — Evaluation (today: 4 boxes → proposed: 3 boxes)

| Today | Proposed |
|---|---|
| `ragas_runner.py` (1 box) | **Keep** — RQ1's core mechanism (RAGAS metrics via the held-constant Qwen2.5 judge). |
| `hallucination_score.py` (1 box) | **Keep** — the composite score is a named research contribution (Objective 4), deserves its own box. |
| `metrics.py` (1 box) | **Keep**, but reposition — this is really shared statistical infrastructure used by *both* the dashboard and the experiment runner, not exclusively part of "the evaluation layer" that runs per-query. Consider drawing it once, with arrows to both the Dashboard and Experiment Runner stages, rather than nested only inside Evaluation. |
| Output-fields box (listing context_relevance, answer_faithfulness, etc.) | **Fold into the `ragas_runner.py` + `hallucination_score.py` captions** rather than a 4th standalone box — it's restating what those two boxes already produce. |

### Stage 5 — Dashboard (today: 6 boxes → proposed: 1–2 boxes)

| Today | Proposed |
|---|---|
| 5 individual page boxes | **1 box: "Streamlit Dashboard (5 views)"** with the 5 names as a bullet caption inside/beside the box, not as 5 separate diagram nodes. This preserves the "5 views" detail you need as evidence for Chapter 1 Objective 5, without spending 5 node-slots drawing it. |
| "Shared components" box | **Optional 2nd box, or fold into the caption** — "metric_cards · latency_chart · heatmap" as a footnote under the main Dashboard box is probably enough; a separate box is only worth it if you want to visually show that components are shared *across* pages (a real, if minor, design point). |

### Stage 6 — Experiments (today: 2 boxes → proposed: 2 boxes, unchanged)

`run_2x2_factorial.py` and `analyse_results.py` are already at the right granularity — one
box each, clearly named, clearly sequential (run the benchmark, then analyse the results).
No change needed here.

### Config (today: 1 box → proposed: 1 box, unchanged, but reposition)

`config/settings.py` currently sits in the "Deployment topology" row. If Diagram A and B
are split (§2), this either disappears from Diagram A entirely (it's plumbing, not a
pipeline stage) or gets drawn as a small dashed box feeding *into* every other stage rather
than sitting alongside `chromadb`/`ragscope` as if it were a service — it configures
everything, it doesn't run as its own component.

### Net effect

| | Today | Proposed |
|---|:---:|:---:|
| Data | 10 | 3 |
| RAG pipeline | 6 | 3 |
| Telemetry | 4 | 2 |
| Evaluation | 4 | 3 |
| Dashboard | 6 | 1–2 |
| Experiments | 2 | 2 |
| Config | 1 | 0–1 |
| **Total** | **~37 (mixed with deployment)** | **~16** (research architecture only; deployment split into its own small optional figure) |

## 4. What this buys you in the viva

With ~16 boxes across 6 clearly labelled, single-purpose stages (Data → Retrieval/Generation
→ Telemetry → Evaluation → Dashboard → Experiments), you can walk an examiner through the
whole system in the time it currently takes to explain just the ingestion sub-stages. Each
box still has a real, defensible answer to "what does this do and why":

1. **Data** — three benchmark corpora, cleaned/chunked/embedded/indexed once.
2. **Retrieval + Generation** — the 2×2 factorial's two independent variables, side by side.
3. **Telemetry** — per-stage timing + tokens, written as one flat record (RQ2's mechanism).
4. **Evaluation** — RAGAS metrics + composite hallucination risk, via a held-constant
   independent judge (RQ1's mechanism).
5. **Dashboard** — real-time and retrospective observability (Objective 5).
6. **Experiments** — the 2×2 factorial run + statistical analysis (Objective 6, RQ3).

That mapping — 6 boxes to 6 objectives/RQs — is itself a strong thing to be able to point
at: "every box on this diagram exists because of a specific research question or SMART
objective," which is a much easier claim to defend than a 37-node diagram where some boxes
(Jupyter, the ingestion manifest, individual page names) don't map to anything in Chapter 1.

## 5. Open questions for you before I redraw anything

1. **Deployment diagram: keep as a small appendix figure, or drop entirely and describe in
   text only?** (My default recommendation: drop it from the main dissertation body; Chapter
   3 §3.6.1 already describes the two-service Docker topology in prose, which is enough.)
2. **`metrics.py` positioning** — draw once with arrows to both Dashboard and Experiments
   (accurate to how it's actually used), or leave nested under Evaluation for simplicity
   even though it's slightly less precise?
3. **Shared dashboard components** — worth a 2nd box, or fold into a caption under the main
   Dashboard box?
4. **Colour-coding** — the current 7-colour legend (teal/purple/amber/coral/blue/green/gray)
   maps one colour per today's-layer. With 6 consolidated stages you could keep the same
   idea with one fewer colour and it'll read cleaner; not a big decision either way.
5. Anything in the [companion analysis document](./architecture_simplification_analysis.md)
   you want actioned *before* I touch the diagram (e.g. removing the dead `pyproject.toml`
   entries first so the "what depends on what" story is already clean when the new diagram
   is drawn), or do you want the diagram redrawn first and the code cleanup done after?

## 6. Next steps (only after you've reviewed both documents)

Once you've told me your calls on the open questions above and which findings from the
analysis document to act on, the sequence I'd suggest is:

1. Redraw `docs/images/ragscope_system_architecture.svg` to the approved ~16-box structure (diagram-only change, no code touched).
2. You review the new diagram.
3. Once the diagram is approved, apply the approved subset of the code/dependency/Docker
   cleanups from the analysis document (each one is independent — you can approve them
   individually, e.g. "yes to the dead pyproject.toml entries, no to the vectorstore
   decorator").
4. Update `docs/architecture.md` (and Chapter 3 if needed) to match whatever was actually
   changed, so the docs and the diagram and the code all stay in sync — the same
   discipline we've been applying throughout this project.

No code will be touched until you've explicitly signed off on step 1 and told me which
items from step 3 to proceed with.
