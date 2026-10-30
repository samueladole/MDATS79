"""
The graph builder's publishing step and the reset tools.

Runs on hand-made extraction records, so neither Ollama nor docling-graph
is needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

nx = pytest.importorskip("networkx")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "graph"))


@pytest.fixture
def builder(tmp_path, monkeypatch):
    import build_graph as bg

    for name, path in {
        "GRAPH_PATH": tmp_path / "graph.json",
        "NODE_CHUNKS_PATH": tmp_path / "node_to_chunks.json",
        "MANIFEST_PATH": tmp_path / "manifest.json",
        "EXTRACTIONS_DIR": tmp_path / "extractions",
    }.items():
        monkeypatch.setattr(bg, name, path)
    (tmp_path / "extractions").mkdir()
    return bg


def _save(bg, chunk_id, entities, ok=True):
    g = nx.DiGraph()
    g.add_node(f"DOC::{chunk_id}", name=chunk_id, __class__="BiomedicalAbstract")
    for e in entities:
        g.add_node(e, name=e.lower(), __class__="ChemicalOrDrug")
        g.add_edge(f"DOC::{chunk_id}", e, label="MENTIONS_COMPOUND")
    rec = ({"chunk_id": chunk_id, "ok": True, "seconds": 10.0,
            "graph": nx.node_link_data(g, edges="edges")} if ok else
           {"chunk_id": chunk_id, "ok": False, "seconds": 3.0, "error": "ValueError: bad json"})
    bg._saved(chunk_id).write_text(json.dumps(rec))


class TestPublish:
    def test_composes_saved_extractions_with_provenance(self, builder):
        _save(builder, "1::chunk_0", ["Aspirin"])
        _save(builder, "2::chunk_0", ["Aspirin", "Ibuprofen"])
        m = builder.publish(["1::chunk_0", "2::chunk_0"], corpus_size=4, label="all")
        node_chunks = json.loads(builder.NODE_CHUNKS_PATH.read_text())
        assert node_chunks["Aspirin"] == ["1::chunk_0", "2::chunk_0"]  # one node, two chunks
        assert m["chunks_ok"] == 2 and m["corpus_coverage_pct"] == 50.0
        assert m["entity_nodes"] == 2 and m["doc_nodes"] == 2

    def test_failures_and_pending_are_counted_not_fatal(self, builder):
        _save(builder, "1::chunk_0", ["Aspirin"])
        _save(builder, "2::chunk_0", [], ok=False)
        m = builder.publish(["1::chunk_0", "2::chunk_0", "3::chunk_0"], corpus_size=3,
                            label="all", force=True)
        assert m["chunks_ok"] == 1 and m["chunks_pending"] == 1
        assert m["failures"] == [{"chunk_id": "2::chunk_0", "error": "ValueError: bad json"}]

    def test_an_unfinished_build_never_replaces_a_larger_live_graph(self, builder):
        builder.GRAPH_PATH.write_text("LIVE")
        builder.MANIFEST_PATH.write_text(json.dumps({"chunks_ok": 298}))
        _save(builder, "1::chunk_0", ["Aspirin"])
        m = builder.publish(["1::chunk_0", "2::chunk_0"], corpus_size=2, label="all")
        assert m["published"] is False
        assert builder.GRAPH_PATH.read_text() == "LIVE"

    def test_a_finished_build_always_publishes(self, builder):
        builder.GRAPH_PATH.write_text("LIVE")
        builder.MANIFEST_PATH.write_text(json.dumps({"chunks_ok": 298}))
        _save(builder, "1::chunk_0", ["Aspirin"])
        builder.publish(["1::chunk_0"], corpus_size=1, label="all")
        assert builder.GRAPH_PATH.read_text() != "LIVE"


class TestClearGraph:
    def test_removes_graph_files_and_keeps_extractions_by_default(self, tmp_path, monkeypatch):
        from config.settings import settings
        from pipeline import maintenance

        graph = tmp_path / "graph.json"
        monkeypatch.setattr(settings, "graph_path", graph)
        monkeypatch.setattr(settings, "graph_node_chunks_path", tmp_path / "node_to_chunks.json")
        for f in ("graph.json", "node_to_chunks.json", "manifest.json"):
            (tmp_path / f).write_text("{}")
        (tmp_path / "extractions").mkdir()
        (tmp_path / "extractions" / "1__chunk_0.json").write_text("{}")

        report = maintenance.clear_graph()
        assert not graph.exists() and not (tmp_path / "manifest.json").exists()
        assert (tmp_path / "extractions").exists() and report.kept

        maintenance.clear_graph(delete_extractions=True)
        assert not (tmp_path / "extractions").exists()


class TestClearDocuments:
    def test_empties_collection_and_removes_bm25_and_manifest(self, tmp_path, monkeypatch):
        from pipeline import maintenance
        from pipeline.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(name="documents", label="My documents", collection="t_docs",
                           bm25_path=tmp_path / "bm25.jsonl",
                           manifest_path=tmp_path / "manifest.json", supports_graph=False)
        kb.bm25_path.write_text("x")
        kb.manifest_path.write_text("{}")

        class Store:
            was_reset = False

            def count(self):
                return 153

            def reset(self):
                Store.was_reset = True

        monkeypatch.setattr(maintenance, "get_knowledge_base", lambda name: kb)
        monkeypatch.setattr("pipeline.vectorstore.get_vector_store", lambda name=None: Store())
        files = tmp_path / "documents"
        files.mkdir()
        monkeypatch.setattr(maintenance.settings, "documents_dir", files)

        report = maintenance.clear_documents()
        assert Store.was_reset
        assert not kb.bm25_path.exists() and not kb.manifest_path.exists()
        assert files.exists() and "153 chunks" in report.removed[0]
