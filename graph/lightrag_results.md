# LightRAG as a GraphRAG baseline

This compares [LightRAG](https://github.com/HKUDS/LightRAG) 1.5.7 with RAGScope's graph retriever. It was run on 27 October 2026 with `graph/lightrag_compare.py`. The raw numbers are in `data/graph/lightrag/build_stats.json` and `compare_results.json`. [How to reproduce](#how-to-reproduce) explains how to rerun it, or try a single question.

**Short answer:** LightRAG does not speed up retrieval.

- **Graph modes:** they call the LLM on every query to extract keywords. That makes them about 140 times slower than our graph retriever on an Apple GPU, and slower still on a CPU.
- **Graph lookup alone:** even with the keywords supplied, it is 3 to 4 times slower than ours.
- **Build:** it takes 3 times as long.

Its value to the paper is as an established GraphRAG baseline, not as a faster engine.

**Provenance.** RAGScope's side of the comparison is the docling-graph graph behind the dissertation's results (Chapter 4, Section 4.9), restricted to the same 50 chunks. The platform still builds its graphs with docling-graph. A different extractor would produce a comparable but not identical graph, so these figures hold for docling-graph specifically.

## Setup

Both systems were given the same input and models, so the comparison isolates the method.

| | RAGScope graph retriever | LightRAG |
|---|---|---|
| Chunks | The first 50 chunks of the 300-chunk sample that the docling-graph builder extracted successfully: 0.16% of the 30,850-chunk corpus. The sample is taken at an even stride across the corpus, with no reference to the benchmark queries (`builder_config.select_sample()`). | The same 50 chunks, one chunk per document |
| Extraction | docling-graph with `llama3.2` (3B) via Ollama, temperature 0, seed 42 | LightRAG's own extraction prompts with the same model and settings |
| Extraction calls per chunk | 1 | 1 (gleaning set to 0; LightRAG's default is 1) |
| Embedder | `all-MiniLM-L6-v2` | Same |
| Storage | NetworkX graph in JSON | LightRAG defaults: JSON key-value store, NanoVectorDB, NetworkX (in memory, local files) |
| Retrieval measured | Graph-only (`fuse_dense=False`), top 5 chunks | `aquery_data` (retrieved context only, no answer), `chunk_top_k=5`, `top_k=40`, no reranker |
| Queries | The 200 benchmark queries | Same |
| Hardware | Apple M5; Ollama on the GPU with its default serving (one request at a time) | Same |

## Build cost

| | RAGScope (docling-graph) | LightRAG |
|---|---|---|
| Seconds per chunk | 11.8 (300-chunk build, `data/graph/manifest.json`) | 35.4 |
| Failed chunks | 2 of 300 | 0 of 50 |
| Graph size on these 50 chunks | 344 nodes (including the 50 chunk nodes), 420 edges | 569 entities, 256 relations |

LightRAG's defaults did not work with a 3B extractor. The first attempt used its defaults: gleaning on, 8 concurrent requests and no output cap. About 200 s per chunk passed, and the first 3 chunks failed. Some extraction calls never emitted LightRAG's completion marker and ran until its 8-minute timeout. Three changes fixed this:

- capping output at 2,048 tokens (complete extractions are about 400);
- turning gleaning off;
- limiting concurrency to 2.

LightRAG's documentation recommends a model of about 30B parameters for extraction, which a CPU laptop cannot run.

## Retrieval latency (200 queries)

| | Median | 95th percentile |
|---|---:|---:|
| **RAGScope graph retriever** (graph-only) | **5.8 ms** | 12.5 ms |
| LightRAG `naive` (vector search over chunks, no graph) | 5.2 ms | 6.1 ms |
| LightRAG `local`, as shipped (LLM keyword call per query) | 822.8 ms | 1,152.6 ms |
| LightRAG `mix`, as shipped (LLM keyword call per query) | 828.0 ms | 1,259.4 ms |
| LightRAG `local`, keywords supplied | 15.2 ms | 27.9 ms |
| LightRAG `mix`, keywords supplied | 24.2 ms | 42.7 ms |

The keyword call ran on the GPU. It was not timed on a CPU. The README's CPU-only measurements put a full answer from the same 3B model at 14–41 s, and the keyword call is a shorter request to that model. So LightRAG's graph modes would take seconds per query on a CPU.

Only LightRAG's `naive` mode matches our latency, and it is plain vector search that does not use the graph.

## What each returns

| | Queries with at least one chunk returned | Mean top-5 overlap with RAGScope |
|---|---:|---:|
| RAGScope graph retriever | 6.0% | |
| LightRAG `naive` | 92.0% | 0.22 |
| LightRAG `local` | 98.5% | 0.20 |
| LightRAG `mix` | 100.0% | 0.20 |

Overlap is computed over the queries where both systems return something.

These rates measure behaviour, not quality. The 50 chunks cover 0.16% of the corpus, so almost no benchmark query has a relevant passage among them.

- **RAGScope's retriever** returns nothing unless the query names an entity in the graph.
- **LightRAG** matches by vector similarity with a low threshold (cosine 0.2). So it nearly always returns its nearest neighbours, relevant or not.

Comparing answer quality needs queries whose answers lie inside the graphed chunks, which this run did not include.

## One question through both systems

The `query` command shows the difference on a single question. This one asks about a disease that one of the 50 chunks describes (output shortened):

```text
Question: What causes Pendred's syndrome?

RAGScope graph retriever (graph-only): 14.0 ms, seed entities: pendred's syndrome
    1. 10902795::chunk_0  [both]

LightRAG naive: 8.6 ms
    1. 10902795::chunk_0  [both]
    2. 14685826::chunk_0
    3. 14871168::chunk_0
    4. 15012892::chunk_0
    5. 15482208::chunk_0

LightRAG local: 779.8 ms (includes the LLM keyword call),
  keywords: {'high_level': ["Pendred's syndrome", 'causes'], 'low_level': ["Pendred's syndrome", 'syndrome']}
    1. 10902795::chunk_0  [both]
    2. 14685826::chunk_0
    ...
```

Both systems rank the Pendred's syndrome abstract first. RAGScope's retriever stops there, because only that chunk mentions the entity. LightRAG fills its five slots with the nearest other chunks, about Schindler disease, blepharospasm, Miller-Fisher syndrome and impetigo. Each LightRAG graph mode also spends about 0.8 s asking the model for keywords.

## Recommendation

- **Keep RAGScope's retriever as the graph strategy.** It is faster to build and to query, and it needs no LLM at query time. That is also why it was designed without one: any gain cannot be attributed to LLM query understanding.
- **Report LightRAG in the paper** as the established GraphRAG baseline, with the build and latency numbers above.
- **For a quality comparison,** build both graphs over the chunks that hold the gold passages for a fixed set of queries, and compare retrieval hit rates on those queries.

## How to reproduce

### Before you start

1. **The 300-chunk sample graph is the live BioASQ graph** (`data/graph/graph.json`). It ships with the prebuilt index (`uv run python scripts/fetch_index.py`), or `uv run python graph/build_graph.py` builds it in about an hour. The 50 chunks are chosen from it. If you have since built the whole corpus, the same chunks are chosen, but the cut-down graph may differ slightly. Restore the sample graph for an exact rerun.
2. **Ollama is running, with `llama3.2` pulled** (`ollama pull llama3.2`), on its default serving: leave `OLLAMA_NUM_PARALLEL` unset. The published timings were measured that way. The script sends LightRAG two requests at a time; more only turned waiting into timeouts.
3. **ChromaDB is reachable.** RAGScope's retriever reads chunk text from the `ragscope_corpus` collection, as set in `.env`.
4. **No install is needed.** LightRAG is not a project dependency: `uv run --with` adds it for the one command. The OpenTelemetry pin keeps the version ChromaDB needs.

### Run the comparison

```bash
# 1. Build LightRAG over the 50 chunks (~30 minutes on an Apple M5)
uv run --with lightrag-hku==1.5.7 --with opentelemetry-api==1.40.0 \
    python graph/lightrag_compare.py build

# 2. Time both retrievers on the 200 benchmark queries (~10 minutes)
uv run --with lightrag-hku==1.5.7 --with opentelemetry-api==1.40.0 \
    python graph/lightrag_compare.py compare

# 3. Print the results as Markdown tables, and save them to share
uv run python graph/lightrag_compare.py report --out lightrag_report.md
```

- `report` prints the build, latency and return-rate tables above, filled from the saved results. It needs neither Ollama nor LightRAG.
- `build --n 100` uses more chunks.
- LightRAG keeps its store in `data/graph/lightrag/store/` and skips chunks already in it. To repeat the build from scratch, delete `data/graph/lightrag/` first. Otherwise the build time it reports is meaningless.
- The 200 queries come from `experiments/results/benchmark_20260813_033105.csv`, which is in the repository.

### Try a single question

```bash
uv run --with lightrag-hku==1.5.7 --with opentelemetry-api==1.40.0 \
    python graph/lightrag_compare.py query "What causes Pendred's syndrome?"
```

This runs the question through RAGScope's graph retriever and through LightRAG's `naive`, `local` and `mix` modes. For each, it prints the time taken, LightRAG's extracted keywords and the top chunks with their opening words. `[both]` marks a chunk the other system also returned. Notes:

- **Modes and depth:** `--modes local mix` picks the LightRAG modes, and `--top-k 3` changes how many chunks each system returns.
- **Build first:** the command needs the LightRAG store from step 1.
- **Cold start:** an untimed warm-up call loads the model first, so the timings match the comparison's.

Only the 50 chunks are searchable. Ask about something they cover, for example amifostine, Pendred's syndrome, HDAC inhibitors or the Notch receptor. `build_stats.json` lists their IDs. A question outside them shows how each system behaves with no answer available: RAGScope's retriever returns nothing, and LightRAG returns its nearest neighbours.

### Outputs

| File | Contents |
|---|---|
| `data/graph/lightrag/build_stats.json` | LightRAG's build: the 50 chunk IDs, time, failures, entities, relations |
| `data/graph/lightrag/compare_results.json` | Every latency, return-rate and overlap figure |
| `data/graph/lightrag/ours_graph.json`, `ours_node_to_chunks.json` | RAGScope's graph cut down to the same 50 chunks |
| `data/graph/lightrag/store/` | LightRAG's own store |

Timings depend on the hardware and on what else the machine is doing, so a rerun will not match the figures above to the millisecond. The ratios between the systems are what the comparison rests on.
