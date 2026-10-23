"""
Tests for GraphRetriever.

Built on a small synthetic graph rather than a real corpus build, so they run
in seconds and do not require docling-graph, Ollama, or a populated ChromaDB.
"""

from __future__ import annotations

import json

import pytest

networkx = pytest.importorskip("networkx")


@pytest.fixture
def graph_artefacts(tmp_path):
    """A 2-chunk graph: aspirin --TREATS--> headache, ibuprofen in another chunk."""
    g = networkx.DiGraph()
    g.add_node("DOC::doc1::chunk_0", name="doc1::chunk_0", __class__="BiomedicalAbstract")
    g.add_node("DOC::doc2::chunk_0", name="doc2::chunk_0", __class__="BiomedicalAbstract")
    g.add_node("Drug_aspirin", name="aspirin", __class__="ChemicalOrDrug")
    g.add_node("Dis_headache", name="headache", __class__="Disease")
    g.add_node("Drug_ibuprofen", name="ibuprofen", __class__="ChemicalOrDrug")
    g.add_edge("DOC::doc1::chunk_0", "Drug_aspirin", label="MENTIONS_COMPOUND")
    g.add_edge("Drug_aspirin", "Dis_headache", label="TREATS")
    g.add_edge("DOC::doc2::chunk_0", "Drug_ibuprofen", label="MENTIONS_COMPOUND")
    g.add_edge("Drug_ibuprofen", "Dis_headache", label="TREATS")

    gp = tmp_path / "graph.json"
    gp.write_text(json.dumps(networkx.node_link_data(g, edges="edges"), default=str))
    ncp = tmp_path / "node_to_chunks.json"
    ncp.write_text(json.dumps({
        "Drug_aspirin": ["doc1::chunk_0"],
        "Dis_headache": ["doc1::chunk_0", "doc2::chunk_0"],
        "Drug_ibuprofen": ["doc2::chunk_0"],
    }))
    return gp, ncp


class _FakeEmbedder:
    """Deterministic 3-dim embedder: dimension 0 fires on 'aspirin'."""

    def _vec(self, text: str) -> list[float]:
        t = text.lower()
        return [1.0 if "aspirin" in t else 0.0,
                1.0 if "headache" in t else 0.0,
                1.0 if "ibuprofen" in t else 0.0]

    def embed_query(self, text):
        return self._vec(text)

    def embed_batch(self, texts):
        return [self._vec(t) for t in texts]


class _FakeStore:
    def get_by_ids(self, ids):
        from pipeline.vectorstore import RetrievedChunk

        return {
            i: RetrievedChunk(chunk_id=i, text=f"text of {i}", score=0.0,
                              metadata={"dataset": "bioasq_factoid"}, score_kind="none")
            for i in ids
        }


def _build(graph_artefacts, monkeypatch, dense_results=()):
    from pipeline.retrieval import graph as graph_mod

    gp, ncp = graph_artefacts

    class _FakeDense:
        def __init__(self, *a, **k):
            pass

        def retrieve(self, query, top_k=None):
            return list(dense_results), {}

    monkeypatch.setattr(graph_mod, "DenseRetriever", _FakeDense)
    return graph_mod.GraphRetriever(
        graph_path=gp, node_chunks_path=ncp,
        vector_store=_FakeStore(), embedding_generator=_FakeEmbedder(), top_k=5,
    )


class TestGraphRetriever:
    def test_registered_as_a_strategy(self):
        from config.settings import RETRIEVAL_STRATEGIES
        from pipeline.retrieval import RETRIEVERS

        assert "graph" in RETRIEVERS
        assert "graph" in RETRIEVAL_STRATEGIES

    def test_missing_artefacts_raise_clearly(self, tmp_path, monkeypatch):
        from pipeline.retrieval.graph import GraphRetriever

        with pytest.raises(FileNotFoundError, match="Graph artefacts missing"):
            GraphRetriever(graph_path=tmp_path / "nope.json",
                           node_chunks_path=tmp_path / "nope2.json",
                           vector_store=_FakeStore(),
                           embedding_generator=_FakeEmbedder())

    def test_document_nodes_are_never_returned_as_seeds(self, graph_artefacts, monkeypatch):
        r = _build(graph_artefacts, monkeypatch)
        _, tel = r.retrieve("aspirin")
        labels = [s["label"] for s in tel["graph_seed_labels"]]
        assert not any(lbl.startswith("doc") for lbl in labels), labels

    def test_seeds_and_traverses_to_source_chunks(self, graph_artefacts, monkeypatch):
        r = _build(graph_artefacts, monkeypatch)
        chunks, tel = r.retrieve("aspirin")
        assert tel["graph_seeds_found"] >= 1
        assert "aspirin" in [s["label"] for s in tel["graph_seed_labels"]]
        assert "doc1::chunk_0" in [c.chunk_id for c in chunks]

    def test_exact_matcher_catches_terms_the_embedder_misses(self, graph_artefacts, monkeypatch):
        """ibuprofen is dimension 2; a query naming it must still seed it."""
        r = _build(graph_artefacts, monkeypatch)
        _, tel = r.retrieve("ibuprofen")
        assert "ibuprofen" in [s["label"] for s in tel["graph_seed_labels"]]

    def test_unseeded_query_degrades_rather_than_crashes(self, graph_artefacts, monkeypatch):
        r = _build(graph_artefacts, monkeypatch)
        chunks, tel = r.retrieve("entirely unrelated quantum chromodynamics")
        assert tel["graph_seeds_found"] == 0
        assert tel["graph_empty_seed"] is True
        assert chunks == []

    def test_telemetry_carries_every_whitelisted_key(self, graph_artefacts, monkeypatch):
        r = _build(graph_artefacts, monkeypatch)
        _, tel = r.retrieve("aspirin")
        for key in ("strategy", "score_kind", "retrieval_ms", "chunks_retrieved",
                    "top_score", "mean_score", "graph_seed_ms", "graph_traverse_ms",
                    "graph_score_ms", "graph_seeds_found", "graph_seed_labels",
                    "graph_subgraph_nodes", "graph_subgraph_edges",
                    "graph_chunks_from_graph_leg"):
            assert key in tel, f"missing telemetry key: {key}"
        assert tel["strategy"] == "graph"

    def test_results_are_marked_rrf_not_cosine(self, graph_artefacts, monkeypatch):
        """Scores are on the RRF scale and must say so, or downstream analysis
        will compare them against dense cosine scores."""
        r = _build(graph_artefacts, monkeypatch)
        chunks, tel = r.retrieve("aspirin")
        assert tel["score_kind"] == "rrf"
        assert all(c.score_kind == "rrf" for c in chunks)

    def test_graph_found_chunks_are_hydrated_with_metadata(self, graph_artefacts, monkeypatch):
        """Chunks the dense leg missed must still carry metadata, or
        build_record logs an empty dataset and the per-dataset breakdown dies."""
        r = _build(graph_artefacts, monkeypatch)
        chunks, _ = r.retrieve("aspirin")
        assert chunks
        assert all(c.metadata.get("dataset") for c in chunks)

    def test_chunks_carry_graph_provenance(self, graph_artefacts, monkeypatch):
        r = _build(graph_artefacts, monkeypatch)
        chunks, _ = r.retrieve("aspirin")
        prov = [c.metadata["graph"] for c in chunks if "graph" in c.metadata]
        assert prov, "no chunk carried graph provenance"
        assert "graph_rank" in prov[0] and "via" in prov[0]


class TestPathGraph:
    """
    The path graph behind the dashboard's interactive retrieval view.

    On the fixture, query "aspirin" seeds aspirin; doc1 is reached through its
    own document node, and doc2 — whose document node lies outside the 2-hop
    subgraph — through headache plus a provenance hop.
    """

    def _path_graph(self, graph_artefacts, monkeypatch, query="aspirin"):
        r = _build(graph_artefacts, monkeypatch)
        chunks, tel = r.retrieve(query)
        return chunks, tel["graph_path"]

    def test_emitted_and_serialisable(self, graph_artefacts, monkeypatch):
        import json

        _, pg = self._path_graph(graph_artefacts, monkeypatch)
        assert pg is not None
        json.dumps(pg)  # must persist through the telemetry store
        assert {"nodes", "edges", "paths", "query"} <= set(pg)

    def test_absent_when_nothing_seeds(self, graph_artefacts, monkeypatch):
        _, pg = self._path_graph(graph_artefacts, monkeypatch,
                                 query="entirely unrelated quantum chromodynamics")
        assert pg is None

    def test_path_edges_form_a_dag(self, graph_artefacts, monkeypatch):
        _, pg = self._path_graph(graph_artefacts, monkeypatch)
        dag = networkx.DiGraph(
            [(e["source"], e["target"]) for e in pg["edges"] if not e["same_layer"]]
        )
        assert networkx.is_directed_acyclic_graph(dag)

    def test_every_path_edge_steps_exactly_one_layer(self, graph_artefacts, monkeypatch):
        """Hop distance strictly increases along a shortest path, which is
        what makes the drawing acyclic. Guard it directly."""
        _, pg = self._path_graph(graph_artefacts, monkeypatch)
        layer = {n["id"]: n["layer"] for n in pg["nodes"]}
        for e in pg["edges"]:
            if e["on_path"]:
                assert layer[e["target"]] == layer[e["source"]] + 1, e

    def test_every_graph_reached_chunk_has_a_path_from_a_seed(self, graph_artefacts, monkeypatch):
        chunks, pg = self._path_graph(graph_artefacts, monkeypatch)
        reached = {c.chunk_id for c in chunks if "graph" in c.metadata}
        assert reached == {p["chunk_id"] for p in pg["paths"]}
        seeds = {n["id"] for n in pg["nodes"] if n["kind"] == "seed"}
        for p in pg["paths"]:
            assert p["nodes"][0] in seeds
            assert p["nodes"][-1] == f"DOC::{p['chunk_id']}"

    def test_chunk_outside_the_subgraph_is_reached_by_a_provenance_hop(
        self, graph_artefacts, monkeypatch
    ):
        _, pg = self._path_graph(graph_artefacts, monkeypatch)
        doc2 = next(p for p in pg["paths"] if p["chunk_id"] == "doc2::chunk_0")
        last = next(e for e in pg["edges"]
                    if e["target"] == "DOC::doc2::chunk_0" and e["on_path"])
        assert last["label"] in ("EXTRACTED_FROM", "MENTIONS_COMPOUND")
        assert doc2["hops"] >= 2

    def test_relations_are_reported_in_their_true_direction(self, graph_artefacts, monkeypatch):
        """Traversal is undirected; the hover must not be."""
        _, pg = self._path_graph(graph_artefacts, monkeypatch)
        relations = {e["relation"] for e in pg["edges"]}
        assert "aspirin —TREATS→ headache" in relations
        # stored as document -> entity, walked as entity -> document
        assert "doc1::chunk_0 —MENTIONS_COMPOUND→ aspirin" in relations

    def test_walking_against_a_relation_is_flagged_reversed(self, graph_artefacts, monkeypatch):
        _, pg = self._path_graph(graph_artefacts, monkeypatch)
        by_pair = {(e["source"], e["target"]): e for e in pg["edges"]}
        # aspirin -> doc1 runs against the stored doc1 -> aspirin MENTIONS edge
        assert by_pair[("Drug_aspirin", "DOC::doc1::chunk_0")]["reversed"] is True
        # aspirin -> headache follows the stored aspirin -> headache TREATS edge
        assert by_pair[("Drug_aspirin", "Dis_headache")]["reversed"] is False

    def test_nodes_carry_pagerank_probability(self, graph_artefacts, monkeypatch):
        _, pg = self._path_graph(graph_artefacts, monkeypatch)
        entity_ppr = [n["ppr"] for n in pg["nodes"] if n["kind"] in ("seed", "entity")]
        assert entity_ppr and all(0.0 <= p <= 1.0 for p in entity_ppr)
        seed = next(n for n in pg["nodes"] if n["kind"] == "seed")
        assert seed["ppr"] > 0 and seed["matcher"] in ("embed", "exact")

    def test_path_build_is_timed_outside_retrieval(self, graph_artefacts, monkeypatch):
        r = _build(graph_artefacts, monkeypatch)
        _, tel = r.retrieve("aspirin")
        assert "graph_path_ms" in tel

    def test_path_graph_is_persisted_by_build_record(self, graph_artefacts, monkeypatch):
        from telemetry.logger import build_record
        from telemetry.token_counter import TokenUsage

        r = _build(graph_artefacts, monkeypatch)
        chunks, tel = r.retrieve("aspirin")
        record = build_record(
            query_id="q1", query="aspirin", dataset="interactive", query_type="factoid",
            llm_model="llama3", retrieval_strategy="graph",
            retrieved_chunks=chunks, retrieval_telemetry=tel,
            answer="a", generation_ms=1.0, e2e_ms=2.0,
            token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1,
                                   total_tokens=2, estimated_cost_usd=0.0),
        )
        assert record["graph_path"] == tel["graph_path"]
