"""
Load your own files (PDF, plain text, Markdown) as passages for ingestion.

Returns the same passage schema as the BioASQ loader, so the files go through
the same cleaning, chunking, embedding and indexing:

    {"id", "text", "title", "dataset": "documents", "source", "page"}

* **PDFs** become one passage per page, so an answer can be traced to its
  page. Only the text layer is read: a scanned PDF with no text layer is
  skipped with a warning (OCR is out of scope here).
* **IDs** come from a hash of the file's content plus the page number, so
  re-ingesting an unchanged file gives the same IDs, and a renamed copy of a
  file is recognised as a duplicate.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from loguru import logger

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def _passage(digest: str, text: str, title: str, source: str, page: int) -> dict:
    return {
        "id": f"doc-{digest[:16]}-p{page}",
        "text": text,
        "title": title,
        "dataset": "documents",
        "source": source,
        "page": page,
    }


def _pdf_passages(path: Path, digest: str, source: str) -> list[dict]:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            reader.decrypt("")  # many "protected" PDFs have an empty user password
        meta_title = (reader.metadata.title if reader.metadata else None) or ""
        title = meta_title.strip() or path.stem
        passages = []
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                passages.append(_passage(digest, text, title, source, number))
    except (PdfReadError, ValueError, KeyError) as exc:  # malformed or still locked
        logger.warning(f"Skipped {source}: could not read PDF ({type(exc).__name__}: {exc}).")
        return []
    if not passages:
        logger.warning(f"Skipped {source}: no text layer (a scanned PDF needs OCR first).")
    return passages


def load_documents(source_dir: Path | str) -> list[dict]:
    """Read every supported file under ``source_dir`` (recursively) as passages."""
    root = Path(source_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Documents folder not found: {root}. Put your PDF, .txt or .md files "
            f"there, or pass another folder with --source."
        )

    files = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
                   and not p.name.startswith("."))
    passages: list[dict] = []
    for path in files:
        source = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix.lower() == ".pdf":
            passages.extend(_pdf_passages(path, digest, source))
            continue
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            passages.append(_passage(digest, text, path.stem, source, 1))
        else:
            logger.warning(f"Skipped {source}: empty file.")

    logger.info(f"Loaded {len(passages):,} passages from {len(files):,} files in {root}.")
    return passages
