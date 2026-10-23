"""
Knowledge-graph retrieval.

Three stages: seed, traverse, hydrate-and-fuse.

1. **Seed** — map the query onto graph entities. Reuses the embedding already
   computed for the dense leg and scores it against every node label by brute
   force, unioned with near-exact string matching, which covers gene symbols
   and acronyms where a 384-dim sentence embedding is weak. The brute-force
   scan is linear in label count and fine at the current scale (~1,600
   labels, ~13 ms); at full-corpus scale it would need an ANN index of its
   own, which is why the cost is reported in ``graph_seed_ms``. Deliberately no LLM on
   this path: it would add hundreds of milliseconds and make it arguable that
   any gain came from LLM query understanding rather than from the graph.

2. **Traverse** — a degree-capped 2-hop ego subgraph around the seeds, then
   Personalized PageRank restricted to that subgraph. Degree capping is the
   hub defence; restricting PPR to the ego subgraph bounds latency, which
   full-graph PageRank in pure Python would not.

3. **Hydrate and fuse** — aggregate node PageRank mass onto the chunks each
   node came from, then fuse that ranking with the dense ranking using the
   same weighted Reciprocal Rank Fusion the hybrid strategy uses. Fusing
   rather than replacing means this strategy differs from hybrid in exactly
   one swapped component, and inherits dense's recall floor so a sparse or
   noisy graph degrades the result instead of destroying it.

The graph is read-only at query time: a NetworkX node-link JSON plus a
node -> chunk_ids index, both produced offline by the builder in ``graph/``.
docling-graph itself is a build-time dependency and is never imported here.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import networkx as nx
from loguru import logger

from config.settings import settings
from pipeline.embeddings import EmbeddingGenerator, get_embedding_generator
from pipeline.retrieval.base import BaseRetriever
from pipeline.retrieval.dense import DenseRetriever
from pipeline.vectorstore import RetrievedChunk, VectorStore, get_vector_store

# Same smoothing constant as hybrid retrieval, so the two are directly
# comparable (Cormack, Clarke and Buettcher, 2009).
RRF_K = 60

#: Expansion through nodes above this degree is skipped. Biomedical graphs have
#: hub entities ("protein", "cancer") that make an uncapped 2-hop neighbourhood
#: reach most of the graph, which destroys relevance.
DEFAULT_DEGREE_CAP = 100

#: The document-root node class written by the builder. These are the chunks
#: themselves, not entities, so they are never returned as seeds.
DOC_PREFIX = "DOC::"

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]{2,}")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _dedupe_labels(seeds: list[dict]) -> list[dict]:
    """One row per distinct surface form, keeping the best-scoring match.

    The builder's alias merging is weak, so the same entity can exist as
    several nodes with identical names. All of them are seeded (recall), but
    reporting each label once keeps the provenance panel readable.
    """
    best: dict[str, dict] = {}
    for s in seeds:
        key = _norm(s["label"])
        if key not in best or s["score"] > best[key]["score"]:
            best[key] = {"label": s["label"], "matcher": s["matcher"], "score": s["score"]}
    return sorted(best.values(), key=lambda h: -h["score"])


class GraphRetriever(BaseRetriever):
    """
    Retrieves chunks by seeding entities from the query and traversing the
    knowledge graph, fused with dense retrieval.

    Parameters
    ----------
    graph_path      : NetworkX node-link JSON written by the graph builder.
    node_chunks_path: ``{node_id: [chunk_id, ...]}`` provenance index.
    vector_store    : VectorStore, for hydrating chunks the dense leg missed.
    embedding_generator : Shared embedder — must be the same model used to
                      build the dense index, or query and label vectors are
                      not in the same space.
    top_k           : Number of final fused results.
    dense_candidates: Dense pool size before fusion.
    graph_candidates: Graph-ranked chunk pool size before fusion.
    dense_weight    : RRF weight for the dense ranking; graph gets 1 - this.
    max_seeds       : Cap on seed entities per query.
    degree_cap      : Skip expansion through nodes above this degree.
    fuse_dense      : Fuse with dense retrieval (default). Set False to return
                      graph-only results, which is what demonstrates the
                      traversal mechanism while graph coverage is partial.
    """

    score_kind = "rrf"

    @property
    def strategy_name(self) -> str:
        return "graph"

    def __init__(
        self,
        graph_path: Path | str | None = None,
        node_chunks_path: Path | str | None = None,
        vector_store: VectorStore | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
        top_k: int | None = None,
        dense_candidates: int = 50,
        graph_candidates: int = 50,
        dense_weight: float | None = None,
        max_seeds: int = 10,
        degree_cap: int = DEFAULT_DEGREE_CAP,
        seed_cos_threshold: float = 0.55,
        fuse_dense: bool = True,
    ) -> None:
        super().__init__(top_k=top_k)
        self._store = vector_store or get_vector_store()
        self._embedder = embedding_generator or get_embedding_generator()
        self._dense = DenseRetriever(
            vector_store=self._store,
            embedding_generator=self._embedder,
            top_k=dense_candidates,
        )
        self.dense_candidates = max(dense_candidates, self.top_k)
        self.graph_candidates = max(graph_candidates, self.top_k)
        self.dense_weight = (
            dense_weight if dense_weight is not None else settings.hybrid_dense_weight
        )
        self.graph_weight = 1.0 - self.dense_weight
        self.max_seeds = max_seeds
        self.degree_cap = degree_cap
        self.seed_cos_threshold = seed_cos_threshold
        # When False the dense leg is skipped and results come from graph
        # traversal alone. Needed because RRF at weights 0.6/0.4 and k=60
        # cannot promote a graph-only chunk into a small top-k: graph rank 1
        # scores 0.4/61 = 0.00656 against dense rank 5 at 0.6/65 = 0.00923.
        # Fused mode therefore lets the graph re-rank what dense already found
        # but never introduce anything, which hides the mechanism entirely
        # while graph coverage is below dense's.
        self.fuse_dense = fuse_dense

        gp = Path(graph_path or settings.graph_path)
        ncp = Path(node_chunks_path or settings.graph_node_chunks_path)
        if not gp.exists() or not ncp.exists():
            raise FileNotFoundError(
                f"Graph artefacts missing ({gp}, {ncp}). Build them first with "
                f"graph/spikes/build_subset.py, or point GRAPH_PATH at an existing graph."
            )

        # Both orientations are kept. Traversal runs on the undirected view, so a
        # relation can be followed against its direction (a protein reaches the
        # drug that inhibits it). The directed original is retained only so the
        # path graph can report each relation the right way round.
        self._digraph: nx.DiGraph = nx.node_link_graph(
            json.loads(gp.read_text()), edges="edges"
        )
        self._graph: nx.Graph = self._digraph.to_undirected()
        self._node_chunks: dict[str, list[str]] = json.loads(ncp.read_text())

        # Entity label index, built once. Document roots are excluded: they are
        # chunks, not entities, and must never be returned as seeds.
        self._labels: list[str] = []
        self._label_nodes: list[str] = []
        for nid, data in self._graph.nodes(data=True):
            if str(nid).startswith(DOC_PREFIX):
                continue
            name = data.get("name") or data.get("label")
            if isinstance(name, str) and name.strip():
                self._labels.append(name.strip())
                self._label_nodes.append(nid)
        self._label_norms = [_norm(x) for x in self._labels]
        self._label_vecs = (
            self._embedder.embed_batch(self._labels) if self._labels else []
        )
        self._degree = dict(self._graph.degree())

        logger.info(
            f"GraphRetriever ready — {self._graph.number_of_nodes()} nodes, "
            f"{self._graph.number_of_edges()} edges, {len(self._labels)} entity labels."
        )

    # ── Stage 1: seeding ──────────────────────────────────────────────────────

    def _seed(self, query: str, query_vec) -> list[dict]:
        """Union of embedding similarity and near-exact matching over labels."""
        hits: dict[str, dict] = {}
        if self._label_vecs is not None and len(self._label_vecs):
            sims = [
                # strict: a dimension mismatch means the label index was built
                # with a different embedder, and must fail rather than truncate.
                sum(a * b for a, b in zip(query_vec, vec, strict=True))
                for vec in self._label_vecs
            ]
            order = sorted(range(len(sims)), key=sims.__getitem__, reverse=True)
            for i in order[: self.max_seeds * 2]:
                if sims[i] >= self.seed_cos_threshold:
                    hits[self._label_nodes[i]] = {
                        "node": self._label_nodes[i], "label": self._labels[i],
                        "matcher": "embed", "score": round(float(sims[i]), 4),
                    }
        # Near-exact token matching: catches gene symbols and acronyms where a
        # general-purpose sentence embedding is weakest.
        for tok in {_norm(t) for t in _TOKEN.findall(query)}:
            if len(tok) < 3:
                continue
            for j, lab_norm in enumerate(self._label_norms):
                if tok == lab_norm:
                    hits.setdefault(self._label_nodes[j], {
                        "node": self._label_nodes[j], "label": self._labels[j],
                        "matcher": "exact", "score": 1.0,
                    })
        return sorted(hits.values(), key=lambda h: -h["score"])[: self.max_seeds]

    # ── Stage 2: traversal ────────────────────────────────────────────────────

    def _traverse(self, seeds: list[dict]) -> tuple[dict[str, float], nx.Graph]:
        """Degree-capped 2-hop ego subgraph, then Personalized PageRank on it."""
        if not seeds:
            return {}, nx.Graph()

        keep: set[str] = set()
        for s in seeds:
            n = s["node"]
            keep.add(n)
            for nb in self._graph.neighbors(n):
                keep.add(nb)
                if self._degree.get(nb, 0) <= self.degree_cap:
                    keep.update(self._graph.neighbors(nb))
        sub = self._graph.subgraph(keep).copy()
        if sub.number_of_nodes() == 0:
            return {}, sub

        personalisation = {s["node"]: float(s["score"]) for s in seeds if s["node"] in sub}
        if not personalisation:
            return {}, sub
        try:
            # max_iter must comfortably exceed ~85. These subgraphs are largely
            # bipartite (entity <-> document), which makes the power iteration
            # oscillate and converge at roughly the damping rate, 0.85 per
            # step; reaching NetworkX's tolerance from there takes ~85 steps.
            # At the previous max_iter=60, 9 of 74 seeded benchmark queries
            # failed to converge and silently fell back to raw seed scores, so
            # every non-seed node received zero probability. The subgraphs are
            # a few dozen nodes, so the extra iterations cost microseconds.
            ppr = nx.pagerank(sub, personalization=personalisation, alpha=0.85, max_iter=200)
        except nx.PowerIterationFailedConvergence:
            logger.warning("PageRank did not converge; falling back to seed scores.")
            ppr = personalisation
        return ppr, sub

    # ── Stage 3: node scores -> chunk ranking ─────────────────────────────────

    def _rank_chunks(
        self, ppr: dict[str, float]
    ) -> tuple[list[tuple[str, float]], dict, dict[str, list[tuple[str, float]]]]:
        """Aggregate node PageRank mass onto the chunks those nodes came from.

        Summed rather than maxed: a chunk mentioning several highly-ranked
        entities is more likely to answer a multi-entity question, which is the
        hypothesis the strategy exists to test. Divided by sqrt(node count) so
        entity-dense chunks cannot win on volume alone.

        Returns the ranking, a human-readable contributor trail per chunk, and
        the contributing node ids with their mass (document nodes included),
        which is what the path graph is built from.
        """
        scores: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        contributors: dict[str, list] = defaultdict(list)
        contributor_ids: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for node, mass in ppr.items():
            nid = str(node)
            if nid.startswith(DOC_PREFIX):
                chunk_ids = [nid[len(DOC_PREFIX):]]
            else:
                chunk_ids = self._node_chunks.get(nid, [])
            label = self._graph.nodes.get(node, {}).get("name", nid)
            is_doc = nid.startswith(DOC_PREFIX)
            for cid in chunk_ids:
                scores[cid] += mass
                counts[cid] += 1
                contributor_ids[cid].append((nid, mass))
                # Document nodes carry PageRank mass but are not entities, so
                # they are excluded from the human-readable "reached via" trail.
                if not is_doc and len(contributors[cid]) < 5:
                    contributors[cid].append({"node": label, "mass": round(mass, 5)})
        normalised = {
            cid: s / math.sqrt(counts[cid]) for cid, s in scores.items() if counts[cid]
        }
        ranked = sorted(normalised.items(), key=lambda kv: -kv[1])[: self.graph_candidates]
        return ranked, contributors, contributor_ids

    # ── Path graph (display only) ─────────────────────────────────────────────

    #: Context entities shown beside the retrieval paths, by PageRank mass.
    PATH_CONTEXT_NODES = 8

    def _relation(self, u: str, v: str) -> tuple[str, str]:
        """The relation between two nodes in its TRUE direction.

        Traversal is undirected, so a path may run against a relation. This
        looks the edge up in the directed graph and reports it as stored, e.g.
        ("INHIBITS", "Cinryze —INHIBITS→ C1-INH") even when the walk went from
        C1-INH to Cinryze.
        """
        name = lambda n: self._display_name(n)  # noqa: E731
        if self._digraph.has_edge(u, v):
            lbl = str(self._digraph.edges[u, v].get("label", "RELATED_TO"))
            return lbl, f"{name(u)} —{lbl}→ {name(v)}"
        if self._digraph.has_edge(v, u):
            lbl = str(self._digraph.edges[v, u].get("label", "RELATED_TO"))
            return lbl, f"{name(v)} —{lbl}→ {name(u)}"
        # No stored edge: the hop is the provenance link from an entity to the
        # chunk it was extracted from (e.g. a protein nested under a drug).
        return "EXTRACTED_FROM", f"{name(u)} was extracted from {name(v)}"

    def _display_name(self, nid: str) -> str:
        nid = str(nid)
        if nid.startswith(DOC_PREFIX):
            return nid[len(DOC_PREFIX):]
        data = self._digraph.nodes.get(nid, {}) if nid in self._digraph else {}
        return str(data.get("name") or data.get("label") or nid)

    def _build_path_graph(
        self,
        query: str,
        seeds: list[dict],
        ppr: dict[str, float],
        sub: nx.Graph,
        results: list[RetrievedChunk],
        graph_ranked: list[tuple[str, float]],
        contributor_ids: dict[str, list[tuple[str, float]]],
    ) -> dict | None:
        """Compact, serialisable graph explaining how each chunk was reached.

        For every result the graph retrieved, the shortest path from any seed
        entity to that chunk is extracted. Nodes are layered by hop distance
        from the seeds and every path edge points outward, so the path edges
        form a directed acyclic graph by construction: hop distance strictly
        increases along a shortest path. A few high-probability context
        entities are added for orientation and flagged as off-path.

        Display only. It is built after the retrieval timers stop, so it never
        contributes to the reported retrieval latency.
        """
        seed_ids = list(dict.fromkeys(s["node"] for s in seeds if s["node"] in sub))
        if not seed_ids:
            return None

        dist = nx.multi_source_dijkstra_path_length(sub, seed_ids)
        graph_scores = dict(graph_ranked)
        seed_info = {s["node"]: s for s in seeds}

        nodes: dict[str, dict] = {}
        edges: dict[tuple[str, str], dict] = {}
        paths: list[dict] = []

        def add_node(nid: str, layer: int) -> None:
            if nid in nodes:
                nodes[nid]["layer"] = min(nodes[nid]["layer"], layer)
                return
            is_doc = nid.startswith(DOC_PREFIX)
            data = self._digraph.nodes.get(nid, {}) if nid in self._digraph else {}
            node = {
                "id": nid,
                "label": self._display_name(nid),
                "kind": "document" if is_doc else ("seed" if nid in seed_info else "entity"),
                "type": "Document" if is_doc else str(data.get("__class__", "Entity")),
                "layer": layer,
                "ppr": round(float(ppr.get(nid, 0.0)), 6),
                "on_path": False,
            }
            if not is_doc:
                node["n_chunks"] = len(self._node_chunks.get(nid, []))
            if nid in seed_info:
                node["matcher"] = seed_info[nid]["matcher"]
                node["match_score"] = round(float(seed_info[nid]["score"]), 4)
            nodes[nid] = node

        def add_edge(u: str, v: str, on_path: bool) -> None:
            key = (u, v)
            if (v, u) in edges:  # already drawn the other way round
                edges[(v, u)]["on_path"] = edges[(v, u)]["on_path"] or on_path
                return
            if key in edges:
                edges[key]["on_path"] = edges[key]["on_path"] or on_path
                return
            label, relation = self._relation(u, v)
            edges[key] = {"source": u, "target": v, "label": label,
                          "relation": relation, "on_path": on_path,
                          # True when the walk ran against the stored relation
                          # (e.g. reached a drug from the protein it inhibits),
                          # so the view can phrase it "inhibited by".
                          "reversed": self._digraph.has_edge(v, u)
                                      and not self._digraph.has_edge(u, v)}

        # Layer 0: the query itself, linked to each seed by how it matched.
        nodes["__query__"] = {
            "id": "__query__", "label": query, "kind": "query", "type": "Query",
            "layer": 0, "ppr": 0.0, "on_path": True,
        }
        for sid in seed_ids:
            add_node(sid, 1)
            nodes[sid]["on_path"] = True
            s = seed_info[sid]
            edges[("__query__", sid)] = {
                "source": "__query__", "target": sid, "label": "SEED",
                "relation": f"query matched '{s['label']}' ({s['matcher']}, {float(s['score']):.3f})",
                "on_path": True,
                "reversed": False,
            }

        # One shortest path per graph-reached result chunk.
        for rank, chunk in enumerate(results, start=1):
            g_meta = chunk.metadata.get("graph")
            if not g_meta:
                continue  # found by the dense leg only; no graph path exists
            cid = chunk.chunk_id
            doc = f"{DOC_PREFIX}{cid}"
            if doc in dist:
                # The chunk's own document node is in the subgraph: walk to it.
                target, tail = doc, []
            else:
                # Otherwise walk to its highest-probability contributing entity,
                # then take the provenance hop into the chunk.
                reachable = [(n, m) for n, m in contributor_ids.get(cid, []) if n in dist]
                if not reachable:
                    continue
                target = max(reachable, key=lambda nm: nm[1])[0]
                tail = [doc]
            _, path = nx.multi_source_dijkstra(sub, seed_ids, target=target)
            path = list(path) + tail

            for i, nid in enumerate(path):
                layer = dist[nid] + 1 if nid in dist else nodes[path[i - 1]]["layer"] + 1
                add_node(nid, layer)
                nodes[nid]["on_path"] = True
            for u, v in zip(path, path[1:], strict=False):
                add_edge(u, v, on_path=True)

            chunk_node = nodes[doc]
            chunk_node.update({
                "kind": "chunk",
                "type": "Retrieved chunk",
                "chunk_id": cid,
                "result_rank": rank,
                "graph_rank": g_meta.get("graph_rank"),
                "graph_score": round(float(graph_scores.get(cid, 0.0)), 6),
                "rrf_score": round(float(chunk.score), 6),
                "from_graph_leg": bool(g_meta.get("from_graph_leg")),
            })
            contributor = path[-2] if len(path) > 1 else path[-1]
            paths.append({
                "chunk_id": cid,
                "result_rank": rank,
                "graph_rank": g_meta.get("graph_rank"),
                "nodes": path,
                "labels": [self._display_name(n) for n in path],
                "hops": len(path) - 1,
                "graph_score": round(float(graph_scores.get(cid, 0.0)), 6),
                "contributor": self._display_name(contributor),
                "contributor_ppr": round(float(ppr.get(contributor, 0.0)), 6),
                "from_graph_leg": bool(g_meta.get("from_graph_leg")),
            })

        # Context: the highest-probability entities that touch the drawn graph.
        on_graph = set(nodes)
        context = sorted(
            (n for n in sub if n not in on_graph and not str(n).startswith(DOC_PREFIX)
             and any(nb in on_graph for nb in sub.neighbors(n))),
            key=lambda n: -ppr.get(n, 0.0),
        )[: self.PATH_CONTEXT_NODES]
        for n in context:
            add_node(n, dist.get(n, 0) + 1)

        # Any remaining subgraph edges between drawn nodes are context, oriented
        # outward by layer so the drawing stays acyclic. Same-layer edges are
        # kept but marked so the view can draw them without an arrowhead.
        drawn = set(nodes) - {"__query__"}
        for u, v in sub.edges():
            if u in drawn and v in drawn:
                lu, lv = nodes[u]["layer"], nodes[v]["layer"]
                a, b = (u, v) if lu <= lv else (v, u)
                add_edge(a, b, on_path=False)
        for e in edges.values():
            e["same_layer"] = nodes[e["source"]]["layer"] == nodes[e["target"]]["layer"]

        return {
            "query": query,
            "nodes": list(nodes.values()),
            "edges": list(edges.values()),
            "paths": paths,
            "subgraph_nodes": sub.number_of_nodes(),
            "probability_note": (
                "Node probability is Personalized PageRank mass within the "
                "query's ego subgraph: the share of time a random walk that "
                "restarts at the seed entities spends at that node. It sums to 1 "
                "across the subgraph, not across the whole graph."
            ),
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> tuple[list[RetrievedChunk], dict]:
        from telemetry.timer import Timer

        k = top_k or self.top_k

        with Timer("embed_query") as embed_t:
            query_vec = self._embedder.embed_query(query)
        with Timer("dense_search") as dense_t:
            dense_chunks = []
            if self.fuse_dense:
                dense_chunks, _ = self._dense.retrieve(query, top_k=self.dense_candidates)
        with Timer("graph_seed") as seed_t:
            seeds = self._seed(query, query_vec)
        with Timer("graph_traverse") as trav_t:
            ppr, sub = self._traverse(seeds)
        with Timer("graph_score") as score_t:
            graph_ranked, contributors, contributor_ids = self._rank_chunks(ppr)

        with Timer("rrf_fusion") as fuse_t:
            rrf: dict[str, float] = defaultdict(float)
            store: dict[str, RetrievedChunk] = {}
            for rank, chunk in enumerate(dense_chunks, start=1):
                rrf[chunk.chunk_id] += self.dense_weight * (1.0 / (RRF_K + rank))
                store[chunk.chunk_id] = chunk
            graph_only_ids = []
            for rank, (cid, _) in enumerate(graph_ranked, start=1):
                rrf[cid] += self.graph_weight * (1.0 / (RRF_K + rank))
                if cid not in store:
                    graph_only_ids.append(cid)
            # Hydrate graph-found chunks with real text and metadata. Without
            # this they would be stubs with empty metadata, which empties the
            # per-dataset breakdown downstream.
            if graph_only_ids:
                store.update(self._store.get_by_ids(graph_only_ids))
            ranked_ids = [c for c in sorted(rrf, key=rrf.__getitem__, reverse=True)
                          if c in store][:k]
            graph_rank_pos = {cid: i for i, (cid, _) in enumerate(graph_ranked)}
            results = []
            for cid in ranked_ids:
                base = store[cid]
                meta = dict(base.metadata)
                if cid in graph_rank_pos:
                    meta["graph"] = {
                        "graph_rank": graph_rank_pos[cid] + 1,
                        "via": contributors.get(cid, []),
                        "from_graph_leg": cid in graph_only_ids,
                    }
                results.append(RetrievedChunk(
                    chunk_id=cid, text=base.text, score=round(rrf[cid], 6),
                    metadata=meta, score_kind="rrf",
                ))

        retrieval_ms = sum(t.elapsed_ms for t in
                           (embed_t, dense_t, seed_t, trav_t, score_t, fuse_t))

        # Built after the retrieval timers have stopped, and timed on its own,
        # so the visualisation never inflates reported retrieval latency.
        with Timer("graph_path") as path_t:
            path_graph = self._build_path_graph(
                query, seeds, ppr, sub, results, graph_ranked, contributor_ids,
            )
        telemetry = {
            "strategy": "graph",
            "score_kind": self.score_kind,
            "fuse_dense": self.fuse_dense,
            "top_k": k,
            "dense_candidates": self.dense_candidates,
            "graph_candidates": self.graph_candidates,
            "dense_weight": self.dense_weight,
            "graph_weight": self.graph_weight,
            "embed_query_ms": embed_t.elapsed_ms,
            "dense_search_ms": dense_t.elapsed_ms,
            "graph_seed_ms": seed_t.elapsed_ms,
            "graph_traverse_ms": trav_t.elapsed_ms,
            "graph_score_ms": score_t.elapsed_ms,
            "rrf_fusion_ms": fuse_t.elapsed_ms,
            "retrieval_ms": round(retrieval_ms, 2),
            "chunks_retrieved": len(results),
            "top_score": results[0].score if results else 0.0,
            "mean_score": (sum(c.score for c in results) / len(results)) if results else 0.0,
            "graph_seeds_found": len(seeds),
            "graph_seed_labels": _dedupe_labels(seeds),
            "graph_empty_seed": not seeds,
            "graph_subgraph_nodes": sub.number_of_nodes(),
            "graph_subgraph_edges": sub.number_of_edges(),
            "graph_candidate_chunks": len(graph_ranked),
            # The number that says whether the graph contributed anything the
            # dense leg had not already found. If this is 0, the strategy has
            # degenerated to dense retrieval with extra steps.
            "graph_chunks_from_graph_leg": sum(
                1 for c in results if c.metadata.get("graph", {}).get("from_graph_leg")
            ),
            # Serialisable explanation of how each chunk was reached, rendered
            # as an interactive graph by the dashboard. None when nothing seeded.
            "graph_path": path_graph,
            "graph_path_ms": path_t.elapsed_ms,
        }
        logger.debug(
            f"Graph retrieval: {len(seeds)} seeds -> {sub.number_of_nodes()} subgraph nodes "
            f"-> {len(results)} chunks in {retrieval_ms:.1f} ms."
        )
        return results, telemetry
