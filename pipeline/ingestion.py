"""
RAGScope — Document Ingestion Pipeline
========================================
Orchestrates the full ingestion workflow:

    Phase 3 of the research project plan (Data Collection and Preparation).

Steps
-----
    1. Load raw passages from the BioASQ corpus
    2. Deduplicate passages across datasets by document ID
    3. Clean and chunk documents using the token-bounded chunker
    4. Generate dense embeddings in streaming mini-batches
       (memory-safe — never holds the full corpus in RAM at once)
    5. Upsert chunks + embeddings into ChromaDB in batches
       (idempotent — safe to re-run; existing chunks are skipped via upsert)
    6. Persist the BM25 corpus index to disk for hybrid retrieval
    7. Write a machine-readable ingestion manifest (JSON)

Design decisions
----------------
* **Streaming batches** — passages are chunked, embedded, and upserted in
  ``embed_batch_size``-chunk windows. Peak RAM is proportional to the batch
  size, not the total corpus. For embed_batch_size=256 with 384-d embeddings
  that is approximately 2 MB of text + 0.4 MB of vectors at any one time.

* **Deduplication** — passage IDs are tracked in a set; any duplicate IDs are
  silently skipped. This prevents ChromaDB from storing redundant embeddings
  and keeps the BM25 index clean. (With a single source corpus this is mostly
  a defensive guard rather than a load-bearing dedup step, but the loader
  interface still returns one dataset per call so the logic is kept general.)

* **Idempotency** — ChromaDB ``upsert`` is idempotent by design. Re-running
  ingestion after a crash will skip already-stored chunks and only add the
  missing ones, without corrupting existing data.

* **Per-dataset statistics** — timing and chunk counts are tracked per dataset
  and written to the ingestion manifest at ``data/ingestion_manifest.json``
  for dissertation reporting (Phase 3 deliverable).

* **BM25 corpus** — persisted as line-delimited JSON (JSONL) via orjson for
  fast incremental writes and streaming reads on large corpora.

* **Dry-run mode** — samples the first 500 passages to estimate total chunk
  counts without writing anything to disk or ChromaDB.

Usage
-----
    # Ingest the BioASQ corpus (recommended for the research experiment)
    uv run python pipeline/ingestion.py --corpus bioasq

    # Wipe the collection and re-ingest from scratch
    uv run python pipeline/ingestion.py --corpus bioasq --reset

    # Dry run: report estimated chunk counts without writing
    uv run python pipeline/ingestion.py --corpus bioasq --dry-run

    # Custom batch sizes (for machines with limited RAM)
    uv run python pipeline/ingestion.py --corpus bioasq --embed-batch-size 64

    # Docker
    docker compose exec ragscope uv run python pipeline/ingestion.py --corpus bioasq
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import orjson
import typer
from loguru import logger
from tqdm import tqdm

from config.settings import settings
from data.loaders import bioasq as bioasq_loader
from data.preprocessing.chunker import Chunk, TokenChunker
from pipeline.embeddings import get_embedding_generator
from pipeline.vectorstore import get_vector_store

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BM25_CORPUS_PATH = PROJECT_ROOT / "data" / "bm25_corpus.jsonl"  # JSONL for streaming reads
MANIFEST_PATH = PROJECT_ROOT / "data" / "ingestion_manifest.json"

# ── Defaults ───────────────────────────────────────────────────────────────────
DEFAULT_EMBED_BATCH = 256  # chunks per embedding forward pass
DEFAULT_UPSERT_BATCH = 500  # chunks per ChromaDB upsert call

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


# ── Statistics dataclasses ─────────────────────────────────────────────────────


@dataclass
class DatasetStats:
    """
    Per-dataset ingestion counters.
    Populated incrementally during the streaming ingest loop.
    """

    name: str
    passages_loaded: int = 0
    passages_skipped: int = 0  # duplicates suppressed by deduplication
    chunks_created: int = 0
    chunks_upserted: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class IngestionSummary:
    """
    Complete ingestion run summary — written to the manifest JSON and
    returned to callers (including the CLI and any programmatic callers
    such as the test suite).
    """

    corpus: str
    started_at: str
    finished_at: str
    total_passages: int
    total_chunks: int
    total_upserted: int
    vector_store_total: int
    bm25_entries: int
    elapsed_seconds: float
    per_dataset: list[dict] = field(default_factory=list)
    dry_run: bool = False

    def to_dict(self) -> dict:
        return {
            "corpus": self.corpus,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_passages": self.total_passages,
            "total_chunks": self.total_chunks,
            "total_upserted": self.total_upserted,
            "vector_store_total": self.vector_store_total,
            "bm25_entries": self.bm25_entries,
            "elapsed_seconds": self.elapsed_seconds,
            "per_dataset": self.per_dataset,
            "dry_run": self.dry_run,
        }


# ── Public API ─────────────────────────────────────────────────────────────────


def ingest_corpus(
    corpus: str = "all",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    embed_batch_size: int = DEFAULT_EMBED_BATCH,
    upsert_batch_size: int = DEFAULT_UPSERT_BATCH,
    reset: bool = False,
    dry_run: bool = False,
) -> IngestionSummary:
    """
    Run the full ingestion pipeline for the specified corpus.

    This is the single callable entry point used by the CLI, tests,
    and any external orchestration scripts.

    Parameters
    ----------
    corpus            : ``"bioasq"`` (only value — kept as an explicit
                        parameter rather than hardcoded so the pipeline can
                        add further corpora without changing this signature)
    chunk_size        : Max tokens per chunk. Defaults to ``settings.chunk_size``.
    chunk_overlap     : Token overlap between consecutive chunks. Defaults to
                        ``settings.chunk_overlap``.
    embed_batch_size  : Number of chunks embedded per forward pass. Controls
                        peak RAM usage. Default 256.
    upsert_batch_size : Number of chunks per ChromaDB upsert call. Default 500.
    reset             : If True, wipe the ChromaDB collection before ingesting.
    dry_run           : If True, run all loading/chunking steps but skip all
                        write operations (ChromaDB, BM25, manifest).

    Returns
    -------
    IngestionSummary
        Complete run statistics, also written to ``data/ingestion_manifest.json``.
    """
    settings.ensure_dirs()
    started_at = datetime.now(UTC).isoformat()
    t_total = time.perf_counter()

    _log_banner(corpus, dry_run)

    # ── Initialise services ────────────────────────────────────────────────────
    store = get_vector_store()
    embedder = get_embedding_generator()
    chunker = TokenChunker(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
    )

    logger.info(
        f"Pipeline config — "
        f"chunk_size={chunker.chunk_size}, "
        f"chunk_overlap={chunker.chunk_overlap}, "
        f"embedding={embedder.model_name} ({embedder.embedding_dim}d), "
        f"embed_batch={embed_batch_size}, "
        f"upsert_batch={upsert_batch_size}"
    )

    if reset and not dry_run:
        logger.warning("--reset: wiping ChromaDB collection …")
        store.reset()
        logger.info("Collection cleared.")

    # ── Stage 1: Load and deduplicate passages ─────────────────────────────────
    all_passages, dataset_stats = _load_all_passages(corpus)

    if not all_passages:
        logger.error(f"No passages loaded for corpus='{corpus}'. Aborting.")
        sys.exit(1)

    logger.info(
        f"Passages loaded (after dedup): {len(all_passages):,}  "
        f"({sum(s.passages_skipped for s in dataset_stats.values()):,} duplicates removed)"
    )

    # ── Dry run: estimate and exit ─────────────────────────────────────────────
    if dry_run:
        _report_dry_run(all_passages, chunker)
        return _build_summary(
            corpus=corpus,
            started_at=started_at,
            all_passages=all_passages,
            dataset_stats=dataset_stats,
            total_chunks=0,
            total_upserted=0,
            store_total=store.count(),
            bm25_entries=0,
            t_total=t_total,
            dry_run=True,
        )

    # ── Stage 2: Stream chunk → embed → upsert ────────────────────────────────
    total_chunks, total_upserted, bm25_entries = _stream_ingest(
        all_passages=all_passages,
        dataset_stats=dataset_stats,
        chunker=chunker,
        embedder=embedder,
        store=store,
        embed_batch_size=embed_batch_size,
        upsert_batch_size=upsert_batch_size,
    )

    # ── Stage 3: Finalise ─────────────────────────────────────────────────────
    summary = _build_summary(
        corpus=corpus,
        started_at=started_at,
        all_passages=all_passages,
        dataset_stats=dataset_stats,
        total_chunks=total_chunks,
        total_upserted=total_upserted,
        store_total=store.count(),
        bm25_entries=bm25_entries,
        t_total=t_total,
        dry_run=False,
    )
    _write_manifest(summary)
    _print_summary_table(summary, dataset_stats)
    return summary


def load_bm25_corpus() -> tuple[list[str], list[str]]:
    """
    Load the persisted BM25 corpus from disk.

    The corpus is stored as JSONL — one ``{"id": "...", "text": "..."}`` object
    per line — which allows streaming reads on large files without loading the
    entire corpus into memory.

    Returns
    -------
    tuple[list[str], list[str]]
        ``(chunk_ids, chunk_texts)`` in the same order, ready to pass directly
        to ``HybridRetriever.build_bm25_index()``.

    Raises
    ------
    FileNotFoundError
        If ingestion has not been run yet.
    """
    if not BM25_CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"BM25 corpus not found at '{BM25_CORPUS_PATH}'.\n"
            "Run ingestion before starting the pipeline:\n"
            "  uv run python pipeline/ingestion.py --corpus all\n"
            "  # Docker:\n"
            "  docker compose exec ragscope uv run python pipeline/ingestion.py --corpus all"
        )

    ids: list[str] = []
    texts: list[str] = []

    with BM25_CORPUS_PATH.open("rb") as fh:
        for raw_line in fh:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            entry = orjson.loads(raw_line)
            ids.append(entry["id"])
            texts.append(entry["text"])

    logger.info(f"BM25 corpus loaded: {len(ids):,} entries.")
    return ids, texts


# ── Stage implementations ──────────────────────────────────────────────────────


def _load_all_passages(
    corpus: str,
) -> tuple[list[dict], dict[str, DatasetStats]]:
    """
    Load passages from all requested datasets and deduplicate.

    Deduplication is by ``passage["id"]`` — a defensive guard rather than
    a load-bearing step now that there is a single source corpus.

    Returns
    -------
    tuple[list[dict], dict[str, DatasetStats]]
    """
    seen_ids: set[str] = set()
    all_passages: list[dict] = []
    stats: dict[str, DatasetStats] = {}

    for ds_name, loader_fn in _resolve_datasets(corpus):
        ds_stat = DatasetStats(name=ds_name)
        t0 = time.perf_counter()

        logger.info(f"[{ds_name}] Loading passages …")
        raw_passages: list[dict] = loader_fn()
        ds_stat.passages_loaded = len(raw_passages)

        for p in raw_passages:
            pid = p.get("id", "")
            if not pid:
                logger.warning(f"[{ds_name}] Passage missing 'id' field — skipping.")
                ds_stat.passages_skipped += 1
                continue
            if pid in seen_ids:
                ds_stat.passages_skipped += 1
                continue
            seen_ids.add(pid)
            all_passages.append(p)

        ds_stat.elapsed_seconds = round(time.perf_counter() - t0, 1)
        stats[ds_name] = ds_stat
        kept = ds_stat.passages_loaded - ds_stat.passages_skipped
        logger.info(
            f"[{ds_name}] {ds_stat.passages_loaded:,} loaded → "
            f"{kept:,} kept, "
            f"{ds_stat.passages_skipped:,} duplicates removed "
            f"({ds_stat.elapsed_seconds}s)"
        )

    return all_passages, stats


def _resolve_datasets(corpus: str) -> list[tuple[str, callable]]:
    """
    Map the ``corpus`` argument to (dataset_name, loader_function) pairs.

    Each loader function returns ``list[dict]`` with the standard passage
    schema: ``{"id": str, "text": str, "title": str, "dataset": str}``.
    """
    all_datasets: list[tuple[str, callable]] = [
        ("bioasq", bioasq_loader.load_corpus),
    ]

    if corpus in ("all", "bioasq"):
        return all_datasets

    logger.error(f"Unknown corpus value '{corpus}'. Valid options: all, bioasq")
    sys.exit(1)


def _stream_ingest(
    all_passages: list[dict],
    dataset_stats: dict[str, DatasetStats],
    chunker: TokenChunker,
    embedder,
    store,
    embed_batch_size: int,
    upsert_batch_size: int,
) -> tuple[int, int, int]:
    """
    Core streaming loop: for each passage, chunk it, accumulate chunks into a
    rolling buffer, and flush the buffer (embed → upsert → write BM25) whenever
    it reaches ``embed_batch_size``.

    The final partial buffer is flushed after the last passage.

    Why flush at ``embed_batch_size`` rather than ``upsert_batch_size``?
    -----------------------------------------------------------------------
    Embedding is the most memory-intensive step (allocates a
    [batch_size × embedding_dim] float matrix). Controlling the embedding
    batch independently of the upsert batch lets users tune RAM usage without
    affecting write throughput.

    The upsert step sub-divides each embedding batch into ``upsert_batch_size``
    slices, so the two batch sizes operate independently.

    Returns
    -------
    tuple[int, int, int]
        (total_chunks_created, total_chunks_upserted, bm25_entries_written)
    """
    total_chunks = 0
    total_upserted = 0
    bm25_entries = 0

    # Open the BM25 JSONL file for incremental appending
    BM25_CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    bm25_fh = BM25_CORPUS_PATH.open("wb")

    # Rolling buffers — cleared after each flush
    chunk_buf: list[Chunk] = []
    dataset_buf: list[str] = []

    logger.info("Streaming ingest started …")

    with tqdm(
        total=len(all_passages),
        desc="Ingesting",
        unit="passage",
        file=sys.stdout,
        dynamic_ncols=True,
        colour="cyan",
    ) as pbar:
        for passage in all_passages:
            ds_name = passage.get("dataset", "unknown")

            # ── Chunk this passage ───────────────────────────────────────────
            new_chunks = chunker.chunk(
                text=passage["text"],
                doc_id=passage["id"],
                metadata={
                    "dataset": ds_name,
                    "title": passage.get("title", ""),
                },
            )

            for chunk in new_chunks:
                chunk_buf.append(chunk)
                dataset_buf.append(ds_name)
                total_chunks += 1

            # ── Flush when the buffer is full ────────────────────────────────
            if len(chunk_buf) >= embed_batch_size:
                n_up, n_bm25 = _flush(
                    chunk_buf,
                    dataset_buf,
                    dataset_stats,
                    embedder,
                    store,
                    upsert_batch_size,
                    bm25_fh,
                )
                total_upserted += n_up
                bm25_entries += n_bm25
                chunk_buf.clear()
                dataset_buf.clear()

            pbar.update(1)
            pbar.set_postfix(chunks=f"{total_chunks:,}", upserted=f"{total_upserted:,}")

        # ── Final flush for the remaining buffer ──────────────────────────────
        if chunk_buf:
            n_up, n_bm25 = _flush(
                chunk_buf,
                dataset_buf,
                dataset_stats,
                embedder,
                store,
                upsert_batch_size,
                bm25_fh,
            )
            total_upserted += n_up
            bm25_entries += n_bm25

    bm25_fh.close()

    logger.info(
        f"Streaming ingest finished — "
        f"{total_chunks:,} chunks created, "
        f"{total_upserted:,} upserted to ChromaDB, "
        f"{bm25_entries:,} BM25 entries written."
    )
    return total_chunks, total_upserted, bm25_entries


def _flush(
    chunk_buf: list[Chunk],
    dataset_buf: list[str],
    dataset_stats: dict[str, DatasetStats],
    embedder,
    store,
    upsert_batch_size: int,
    bm25_fh,
) -> tuple[int, int]:
    """
    Flush one buffer:
        1. Embed all chunks in one vectorised call.
        2. Upsert to ChromaDB in ``upsert_batch_size`` sub-batches.
        3. Append each chunk to the BM25 JSONL file.
        4. Update per-dataset chunk counters.

    Returns
    -------
    tuple[int, int]
        (chunks_upserted, bm25_lines_written)
    """
    texts = [c.text for c in chunk_buf]
    embeddings = embedder.embed_batch(texts)

    # Upsert to ChromaDB (sub-batches handled inside store.add_chunks)
    n_upserted = store.add_chunks(chunk_buf, embeddings, batch_size=upsert_batch_size)

    # Write to BM25 JSONL — one compact JSON line per chunk
    for chunk in chunk_buf:
        bm25_fh.write(orjson.dumps({"id": chunk.chunk_id, "text": chunk.text}) + b"\n")

    # Update per-dataset statistics
    for chunk, ds_name in zip(chunk_buf, dataset_buf):
        if ds_name in dataset_stats:
            dataset_stats[ds_name].chunks_created += 1
            dataset_stats[ds_name].chunks_upserted += 1

    return n_upserted, len(chunk_buf)


# ── Utilities ──────────────────────────────────────────────────────────────────


def _report_dry_run(all_passages: list[dict], chunker: TokenChunker) -> None:
    """
    Sample the first 500 passages, measure actual chunk counts, and extrapolate
    an estimate for the full corpus. Writes nothing to disk.
    """
    sample_n = min(500, len(all_passages))
    sample = all_passages[:sample_n]

    sample_chunks = sum(
        len(chunker.chunk(p["text"], doc_id=p["id"]))
        for p in tqdm(
            sample, desc="Sampling (dry run)", unit="passage", file=sys.stdout, dynamic_ncols=True
        )
    )
    avg = sample_chunks / sample_n if sample_n else 0
    est = int(avg * len(all_passages))
    logger.info(
        f"Dry-run estimate: "
        f"{len(all_passages):,} passages → "
        f"~{est:,} chunks "
        f"(avg {avg:.1f} chunks/passage based on {sample_n}-passage sample)"
    )
    logger.info("No data written. Remove --dry-run to run for real.")


def _build_summary(
    corpus: str,
    started_at: str,
    all_passages: list[dict],
    dataset_stats: dict[str, DatasetStats],
    total_chunks: int,
    total_upserted: int,
    store_total: int,
    bm25_entries: int,
    t_total: float,
    dry_run: bool,
) -> IngestionSummary:
    return IngestionSummary(
        corpus=corpus,
        started_at=started_at,
        finished_at=datetime.now(UTC).isoformat(),
        total_passages=len(all_passages),
        total_chunks=total_chunks,
        total_upserted=total_upserted,
        vector_store_total=store_total,
        bm25_entries=bm25_entries,
        elapsed_seconds=round(time.perf_counter() - t_total, 1),
        per_dataset=[
            {
                "dataset": s.name,
                "passages_loaded": s.passages_loaded,
                "passages_skipped": s.passages_skipped,
                "chunks_created": s.chunks_created,
                "chunks_upserted": s.chunks_upserted,
            }
            for s in dataset_stats.values()
        ],
        dry_run=dry_run,
    )


def _write_manifest(summary: IngestionSummary) -> None:
    """Persist the ingestion manifest as a formatted JSON file."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_bytes(orjson.dumps(summary.to_dict(), option=orjson.OPT_INDENT_2))
    logger.info(f"Manifest → {MANIFEST_PATH}")


def _print_summary_table(
    summary: IngestionSummary,
    dataset_stats: dict[str, DatasetStats],
) -> None:
    """Print a formatted summary to the logger."""
    sep = "─" * 64
    logger.info(sep)
    logger.info("  INGESTION SUMMARY")
    logger.info(sep)
    logger.info(f"  Corpus            : {summary.corpus}")
    logger.info(f"  Total passages    : {summary.total_passages:>10,}")
    logger.info(f"  Total chunks      : {summary.total_chunks:>10,}")
    logger.info(f"  Chunks upserted   : {summary.total_upserted:>10,}")
    logger.info(f"  Vector store size : {summary.vector_store_total:>10,} (cumulative)")
    logger.info(f"  BM25 entries      : {summary.bm25_entries:>10,}")
    logger.info(f"  Wall-clock time   : {summary.elapsed_seconds:>9.1f}s")
    logger.info(sep)
    logger.info("  Per-dataset:")
    logger.info(
        f"    {'Dataset':<22}  {'Passages':>8}  {'Deduped':>7}  {'Chunks':>8}  {'Upserted':>8}"
    )
    logger.info(f"    {'─' * 22}  {'─' * 8}  {'─' * 7}  {'─' * 8}  {'─' * 8}")
    for s in dataset_stats.values():
        logger.info(
            f"    {s.name:<22}  {s.passages_loaded:>8,}  "
            f"{s.passages_skipped:>7,}  {s.chunks_created:>8,}  {s.chunks_upserted:>8,}"
        )
    logger.info(sep)
    logger.info(f"  Manifest  → {MANIFEST_PATH}")
    logger.info(f"  BM25 idx  → {BM25_CORPUS_PATH}")
    logger.info(sep)


def _log_banner(corpus: str, dry_run: bool) -> None:
    sep = "═" * 64
    mode = " [DRY RUN — no writes]" if dry_run else ""
    logger.info(sep)
    logger.info(f"  RAGScope — Document Ingestion Pipeline{mode}")
    logger.info(f"  corpus={corpus}")
    logger.info(sep)


# ── CLI ────────────────────────────────────────────────────────────────────────


@app.command()
def main(
    corpus: str = typer.Option(
        "bioasq",
        help="Corpus to ingest: bioasq (or 'all', an alias for the same single corpus).",
    ),
    chunk_size: int = typer.Option(
        512,
        help="Maximum tokens per chunk (must match the value used at query time).",
    ),
    chunk_overlap: int = typer.Option(
        64,
        help="Token overlap between consecutive chunks.",
    ),
    embed_batch_size: int = typer.Option(
        DEFAULT_EMBED_BATCH,
        help=(
            "Chunks embedded per forward pass. "
            "Reduce if you hit out-of-memory errors (e.g. --embed-batch-size 64). "
            "Increase for faster ingestion on GPU (e.g. --embed-batch-size 512)."
        ),
    ),
    upsert_batch_size: int = typer.Option(
        DEFAULT_UPSERT_BATCH,
        help="Chunks per ChromaDB upsert call.",
    ),
    reset: bool = typer.Option(
        False,
        "--reset",
        help=(
            "Delete and recreate the ChromaDB collection before ingesting. "
            "USE WITH CAUTION — this permanently erases all stored embeddings."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=(
            "Load and chunk data but skip all write operations. "
            "Prints estimated chunk counts. Useful for validating config "
            "before committing to a long ingest run."
        ),
    ),
) -> None:
    """
    Ingest benchmark corpora into ChromaDB and build the BM25 index.

    Run this once after downloading datasets. Subsequent runs are idempotent —
    existing chunks will be skipped via ChromaDB upsert semantics.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:HH:mm:ss} | {level:<8} | {message}",
        colorize=True,
    )

    ingest_corpus(
        corpus=corpus,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_batch_size=embed_batch_size,
        upsert_batch_size=upsert_batch_size,
        reset=reset,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    app()
