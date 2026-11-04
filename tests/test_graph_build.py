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

    def test_never_replaces_a_graph_with_an_empty_one(self, builder):
        builder.GRAPH_PATH.write_text("LIVE")
        builder.MANIFEST_PATH.write_text(json.dumps({"chunks_ok": 298}))
        m = builder.publish(["1::chunk_0", "2::chunk_0"], corpus_size=2, label="sample",
                            force=True)  # --publish-only with nothing saved
        assert m["published"] is False
        assert builder.GRAPH_PATH.read_text() == "LIVE"

    def test_a_finished_build_always_publishes(self, builder):
        builder.GRAPH_PATH.write_text("LIVE")
        builder.MANIFEST_PATH.write_text(json.dumps({"chunks_ok": 298}))
        _save(builder, "1::chunk_0", ["Aspirin"])
        builder.publish(["1::chunk_0"], corpus_size=1, label="all")
        assert builder.GRAPH_PATH.read_text() != "LIVE"


class TestSample:
    def test_even_stride_length_filter_and_file_order(self, tmp_path):
        import builder_config

        corpus = tmp_path / "corpus.jsonl"
        # 300 rows -> stride 18; every third sampled row is too short to keep.
        rows = [{"id": f"{i}::chunk_0", "text": ("short" if i % 54 == 0 else "x" * 300)}
                for i in range(300)]
        corpus.write_text("".join(json.dumps(r) + "\n" for r in rows))
        picked = builder_config.select_sample(10, corpus)
        assert len(picked) == 10
        assert all(200 < len(text) < 2000 for _, text in picked)
        stems = [cid.replace("::", "__") for cid, _ in picked]
        assert stems == sorted(stems)
        assert picked == builder_config.select_sample(10, corpus)  # deterministic


class TestKnowledgeBaseScope:
    def test_documents_build_uses_its_own_corpus_and_folder(self, tmp_path, monkeypatch):
        import build_graph as bg

        from pipeline.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(
            name="documents", label="My documents", collection="t_docs",
            bm25_path=tmp_path / "bm25_documents.jsonl", manifest_path=tmp_path / "m.json",
            graph_path=tmp_path / "graph" / "documents" / "graph.json",
            node_chunks_path=tmp_path / "graph" / "documents" / "node_to_chunks.json",
        )
        kb.bm25_path.write_text(json.dumps({"id": "doc-1-p1::chunk_0", "text": "Aspirin."}) + "\n")
        monkeypatch.setattr("pipeline.knowledge_base.get_knowledge_base", lambda name: kb)
        for name in ("CORPUS_PATH", "GRAPH_PATH", "NODE_CHUNKS_PATH", "MANIFEST_PATH",
                     "EXTRACTIONS_DIR", "WORK_DIR"):
            monkeypatch.setattr(bg, name, getattr(bg, name))  # restored after the test
        bg.use_knowledge_base("documents")
        assert kb.graph_path == bg.GRAPH_PATH
        assert kb.graph_dir / "extractions" == bg.EXTRACTIONS_DIR
        assert bg.target_chunks(all_chunks=True) == [("doc-1-p1::chunk_0", "Aspirin.")]


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
                           manifest_path=tmp_path / "manifest.json",
                           graph_path=tmp_path / "graph" / "graph.json",
                           node_chunks_path=tmp_path / "graph" / "node_to_chunks.json")
        kb.bm25_path.write_text("x")
        kb.manifest_path.write_text("{}")
        kb.graph_dir.mkdir()
        kb.graph_path.write_text("{}")
        (kb.graph_dir / "extractions").mkdir()
        compared = tmp_path / "lightrag" / "documents"
        compared.mkdir(parents=True)

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
        assert not kb.graph_dir.exists(), "the documents graph goes with its chunks"
        assert not compared.exists(), "so does a LightRAG comparison over them"
        assert files.exists() and "153 chunks" in report.removed[0]


class TestBackgroundBuild:
    @pytest.fixture
    def kb(self, tmp_path):
        from pipeline.knowledge_base import KnowledgeBase

        return KnowledgeBase(
            name="documents", label="My documents", collection="t_docs",
            bm25_path=tmp_path / "bm25.jsonl", manifest_path=tmp_path / "m.json",
            graph_path=tmp_path / "graph" / "graph.json",
            node_chunks_path=tmp_path / "graph" / "node_to_chunks.json",
        )

    def test_a_stale_pid_file_is_not_a_running_build(self, kb):
        from pipeline import graph_jobs

        kb.graph_dir.mkdir()
        (kb.graph_dir / "build.pid").write_text("999999999")
        assert graph_jobs.running_pid(kb) is None
        assert graph_jobs.stop_build(kb) is False

    def test_progress_counts_saved_extractions_for_this_corpus(self, kb):
        from pipeline import graph_jobs

        kb.bm25_path.write_text("".join(json.dumps({"id": f"d{i}::chunk_0", "text": "t"}) + "\n"
                                        for i in range(4)))
        (kb.graph_dir / "extractions").mkdir(parents=True)
        for name in ("d0__chunk_0", "d1__chunk_0", "removed-file__chunk_0"):
            (kb.graph_dir / "extractions" / f"{name}.json").write_text("{}")
        status = graph_jobs.build_status(kb)
        assert (status.extracted, status.total, status.running) == (2, 4, False)

    def test_explains_what_is_missing_instead_of_starting(self, kb, monkeypatch):
        from pipeline import graph_jobs

        monkeypatch.setattr(graph_jobs, "docling_installed", lambda: False)
        assert "uv sync --extra graph" in graph_jobs.start_build(kb)
        monkeypatch.setattr(graph_jobs, "docling_installed", lambda: True)
        assert "Index the documents first" in graph_jobs.start_build(kb)


def test_extraction_goes_to_the_configured_ollama(tmp_path):
    """docling-graph ignores OLLAMA_BASE_URL, so the builder must pass it explicitly."""
    import os

    import builder_config

    conf = builder_config.build_config(tmp_path / "chunk.md")
    assert conf["llm_overrides"]["connection"]["base_url"] == os.environ["OLLAMA_BASE_URL"]
