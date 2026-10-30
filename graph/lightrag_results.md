# LightRAG as a GraphRAG baseline

Comparison of [LightRAG](https://github.com/HKUDS/LightRAG) 1.5.7 with RAGScope's docling-graph retriever, run on 27 October 2026 with `graph/lightrag_compare.py`. Raw numbers: `data/graph/lightrag/build_stats.json` and `compare_results.json`.

**Short answer:** LightRAG does not speed up retrieval. Its graph modes make an LLM call on every query to extract keywords, which makes them about 140 times slower than our graph retriever on an Apple GPU, and slower still on a CPU. With the keywords supplied, its graph lookup alone is 3 to 4 times slower than ours. It also takes 3 times as long to build. Its value to the paper is as an established GraphRAG baseline, not as a faster engine.

## Setup

Both systems were given the same input and models, so the comparison isolates the method.

| | RAGScope graph retriever | LightRAG |
|---|---|---|
| Chunks | The first 50 chunks of the 300-chunk corpus-principled sample in `data/graph/sample/` (0.16% of the 30,850-chunk corpus) | The same 50 chunks, one chunk per document |
| Extraction model | `llama3.2` (3B) via Ollama, temperature 0, seed 42 | Same |
| Extraction calls per chunk | 1 | 1 (gleaning set to 0; LightRAG's default is 1) |
| Embedder | `all-MiniLM-L6-v2` | Same |
| Storage | NetworkX graph in JSON | LightRAG defaults: JSON key-value store, NanoVectorDB, NetworkX (in memory, local files) |
| Retrieval measured | Graph-only (`fuse_dense=False`), top 5 chunks | `aquery_data` (retrieved context only, no answer), `chunk_top_k=5`, `top_k=40`, no reranker |
| Queries | The 200 benchmark queries | Same |
| Hardware | Apple M5; Ollama on the GPU with `OLLAMA_NUM_PARALLEL=1` | Same |

## Build cost

| | RAGScope (docling-graph) | LightRAG |
|---|---|---|
| Seconds per chunk | 11.8 (300-chunk build, `data/graph/manifest.json`) | 35.4 |
| Failed chunks | 2 of 300 | 0 of 50 |
| Graph size on these 50 chunks | 344 nodes (including the 50 chunk nodes), 420 edges | 569 entities, 256 relations |

LightRAG's defaults did not work with a 3B extractor. In the first attempt, with gleaning on, 8 concurrent requests and no output cap, about 200 s per chunk passed and the first 3 chunks failed. Some extraction calls never emitted LightRAG's completion marker and ran until its 8-minute timeout. Capping output at 2,048 tokens (complete extractions are about 400), turning gleaning off and limiting concurrency to 2 fixed this. LightRAG's documentation recommends a model of about 30B parameters for extraction, which a CPU laptop cannot run.

## Retrieval latency (200 queries)

| | Median | 95th percentile |
|---|---:|---:|
| **RAGScope graph retriever** (graph-only) | **5.8 ms** | 12.5 ms |
| LightRAG `naive` (vector search over chunks, no graph) | 5.2 ms | 6.1 ms |
| LightRAG `local`, as shipped (LLM keyword call per query) | 822.8 ms | 1,152.6 ms |
| LightRAG `mix`, as shipped (LLM keyword call per query) | 828.0 ms | 1,259.4 ms |
| LightRAG `local`, keywords supplied | 15.2 ms | 27.9 ms |
| LightRAG `mix`, keywords supplied | 24.2 ms | 42.7 ms |

The keyword call ran on the GPU. It was not timed on a CPU, but the README's CPU-only measurements put a full answer from the same 3B model at 14–41 s. The keyword call is a shorter request to that model, so LightRAG's graph modes would take seconds per query on a CPU.

## What each returns

| | Queries with at least one chunk returned | Mean top-5 overlap with RAGScope |
|---|---:|---:|
| RAGScope graph retriever | 6.0% | |
| LightRAG `naive` | 92.0% | 0.22 |
| LightRAG `local` | 98.5% | 0.20 |
| LightRAG `mix` | 100.0% | 0.20 |

Overlap is computed over the queries where both systems return something.

These rates measure behaviour, not quality. The 50 chunks cover 0.16% of the corpus, so almost no benchmark query has a relevant passage among them. Our retriever returns nothing unless the query names an entity in the graph. LightRAG matches by vector similarity with a low threshold (cosine 0.2), so it nearly always returns its nearest neighbours, relevant or not. Comparing answer quality needs queries whose answers lie inside the graphed chunks, which this run did not include.

## Recommendation

- Keep RAGScope's retriever as the graph strategy. It is faster to build and to query, and it needs no LLM at query time. That is also why it was designed without one: any gain cannot be attributed to LLM query understanding.
- Report LightRAG in the paper as the established GraphRAG baseline, with the build and latency numbers above.
- For a quality comparison, build both graphs over the chunks that hold the gold passages for a fixed query subset, and compare retrieval hit rate on those queries.
