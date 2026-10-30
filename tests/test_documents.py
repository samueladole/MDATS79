"""
Ingesting your own files: the loader, and that they stay out of the benchmark index.

No embedding model or vector store is used: both are replaced with fakes.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pypdf")


def _text_pdf(path, lines: list[str]) -> None:
    """Write a minimal one-page PDF with a real text layer."""
    content = "BT /F1 12 Tf 72 720 Td " + " ".join(
        f"({line}) Tj 0 -16 Td" for line in lines
    ) + " ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(out)


def _blank_pdf(path) -> None:
    """A PDF with a page but no text layer, like a scan."""
    from pypdf import PdfWriter

    w = PdfWriter()
    w.add_blank_page(width=612, height=792)
    with path.open("wb") as fh:
        w.write(fh)


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "docs"
    (d / "sub").mkdir(parents=True)
    (d / "notes.txt").write_text("Aspirin inhibits COX-1 and COX-2.")
    (d / "sub" / "summary.md").write_text("# Metformin\nFirst-line therapy for type 2 diabetes.")
    _text_pdf(d / "paper.pdf", ["Empagliflozin inhibits SGLT2.", "It lowers blood glucose."])
    _blank_pdf(d / "scan.pdf")
    (d / "ignored.docx").write_bytes(b"not supported")
    return d


class TestLoader:
    def test_reads_text_markdown_and_pdf_pages(self, folder):
        from data.loaders.documents import load_documents

        passages = load_documents(folder)
        by_source = {p["source"]: p for p in passages}
        assert set(by_source) == {"notes.txt", "sub/summary.md", "paper.pdf"}
        assert "SGLT2" in by_source["paper.pdf"]["text"]
        assert by_source["paper.pdf"]["page"] == 1
        assert all(p["dataset"] == "documents" for p in passages)

    def test_scanned_pdf_is_skipped_not_fatal(self, folder):
        from data.loaders.documents import load_documents

        assert "scan.pdf" not in {p["source"] for p in load_documents(folder)}

    def test_ids_are_stable_and_follow_content(self, folder):
        from data.loaders.documents import load_documents

        first = {p["source"]: p["id"] for p in load_documents(folder)}
        (folder / "copy.txt").write_text((folder / "notes.txt").read_text())
        second = {p["source"]: p["id"] for p in load_documents(folder)}
        assert second["notes.txt"] == first["notes.txt"]
        assert second["copy.txt"] == first["notes.txt"]  # same content, same id

    def test_missing_folder_says_where_to_put_files(self, tmp_path):
        from data.loaders.documents import load_documents

        with pytest.raises(FileNotFoundError, match="Documents folder not found"):
            load_documents(tmp_path / "nope")


class _FakeStore:
    def __init__(self):
        self.chunks, self.was_reset = [], False

    def reset(self):
        self.was_reset = True

    def add_chunks(self, chunks, embeddings, batch_size=500):
        self.chunks.extend(chunks)
        return len(chunks)

    def count(self):
        return len(self.chunks)


class _FakeEmbedder:
    model_name, embedding_dim = "fake", 3

    def embed_batch(self, texts):
        return [[0.0, 0.0, 1.0] for _ in texts]


class TestDocumentsIngestion:
    def test_goes_to_its_own_collection_and_bm25_file(self, folder, tmp_path, monkeypatch):
        import pipeline.ingestion as ing
        from pipeline.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(
            name="documents", label="My documents", collection="test_documents",
            bm25_path=tmp_path / "bm25_documents.jsonl",
            manifest_path=tmp_path / "documents_manifest.json", supports_graph=False,
        )
        requested = []
        store = _FakeStore()
        monkeypatch.setattr(ing, "get_knowledge_base", lambda name: kb)
        monkeypatch.setattr(ing, "get_vector_store",
                            lambda name=None: requested.append(name) or store)
        monkeypatch.setattr(ing, "get_embedding_generator", lambda: _FakeEmbedder())

        summary = ing.ingest_corpus("documents", source=folder)

        assert requested == ["test_documents"]
        assert store.was_reset, "the documents collection must mirror its folder"
        assert summary.total_chunks == len(store.chunks) == 3
        assert kb.bm25_path.exists() and kb.manifest_path.exists()
        meta = {c.metadata["source"]: c.metadata for c in store.chunks}
        assert meta["paper.pdf"]["page"] == 1

    def test_bioasq_and_documents_never_share_storage(self):
        from pipeline.knowledge_base import get_knowledge_base

        bio, docs = get_knowledge_base("bioasq"), get_knowledge_base("documents")
        assert bio.collection != docs.collection
        assert bio.bm25_path != docs.bm25_path
        assert bio.manifest_path != docs.manifest_path

    def test_graph_retrieval_is_refused_for_documents(self):
        from pipeline.rag import RAGPipeline

        with pytest.raises(ValueError, match="Graph retrieval needs a knowledge graph"):
            RAGPipeline(retrieval_strategy="graph", knowledge_base="documents",
                        run_evaluation=False)


def test_empty_folder_leaves_the_collection_alone(tmp_path, monkeypatch):
    import pipeline.ingestion as ing

    store = _FakeStore()
    monkeypatch.setattr(ing, "get_vector_store", lambda name=None: store)
    monkeypatch.setattr(ing, "get_embedding_generator", lambda: _FakeEmbedder())
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="No readable text"):
        ing.ingest_corpus("documents", source=tmp_path / "empty")
    assert not store.was_reset
