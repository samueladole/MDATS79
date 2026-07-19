"""
RAGScope — Download All Datasets
==================================
One-shot script that pre-downloads and caches all benchmark
datasets to ``data/raw/``.

Usage
-----
    # Parallel mode (default)
    uv run python data/loaders/download_all.py

    # Explicit parallel mode
    uv run python data/loaders/download_all.py --parallel

    # Sequential mode
    uv run python data/loaders/download_all.py --sequential

    # Limit worker count in parallel mode
    uv run python data/loaders/download_all.py --workers 2
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from loguru import logger

# ── Logging setup ─────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")


def download_msmarco() -> None:
    """Download MS MARCO v2.1 (train split — the only split the loaders read)."""
    logger.info("Downloading MS MARCO v2.1 (train split) …")

    load_dataset("microsoft/ms_marco", "v2.1", split="train", cache_dir="data/raw/msmarco")

    logger.info("MS MARCO downloaded. ✓")


def download_natural_questions() -> None:
    """Download Natural Questions (dev config, validation split)."""
    logger.info("Downloading Natural Questions (validation split) …")

    load_dataset(
        "google-research-datasets/natural_questions",
        "dev",
        split="validation",
        cache_dir="data/raw/natural_questions",
    )

    logger.info("Natural Questions downloaded. ✓")


def download_hotpotqa() -> None:
    """Download HotpotQA fullwiki (validation split)."""
    logger.info("Downloading HotpotQA fullwiki (validation split) …")

    load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="validation", cache_dir="data/raw/hotpotqa")

    logger.info("HotpotQA downloaded. ✓")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--parallel",
        action="store_true",
        help="Run downloads in parallel (default).",
    )
    mode.add_argument(
        "--sequential",
        action="store_true",
        help="Run downloads sequentially.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of worker threads in parallel mode (default: 3).",
    )

    return parser.parse_args()


def run_sequential(jobs: dict[str, Callable[[], None]]) -> dict[str, bool]:
    """Run download jobs sequentially."""

    results: dict[str, bool] = {}

    for name, fn in jobs.items():
        try:
            fn()
            results[name] = True
        except Exception as exc:
            logger.error(f"Failed to download {name}: {exc}")
            results[name] = False

    return results


def run_parallel(jobs: dict[str, Callable[[], None]], workers: int) -> dict[str, bool]:
    """Run download jobs in parallel using a thread pool."""

    results: dict[str, bool] = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_name = {executor.submit(fn): name for name, fn in jobs.items()}

        for future in as_completed(future_to_name):
            name = future_to_name[future]

            try:
                future.result()
                results[name] = True
            except Exception as exc:
                logger.error(f"Failed to download {name}: {exc}")
                results[name] = False

    return results


def main() -> None:
    """Main entry point."""

    args = parse_args()

    logger.info("=" * 60)
    logger.info("RAGScope — Dataset Downloader")
    logger.info("=" * 60)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    data_dir = PROJECT_ROOT / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    jobs = {
        "MS MARCO": download_msmarco,
        "Natural Questions": download_natural_questions,
        "HotpotQA": download_hotpotqa,
    }

    # Parallel by default unless explicitly sequential
    use_parallel = not args.sequential

    if use_parallel:
        logger.info(f"Mode: PARALLEL ({args.workers} workers)")
        results = run_parallel(jobs, args.workers)
    else:
        logger.info("Mode: SEQUENTIAL")
        results = run_sequential(jobs)

    logger.info("-" * 60)
    for name, _ in jobs.items():
        ok = results.get(name, False)
        status = "✓" if ok else "✗ FAILED"
        logger.info(f"  {name:<25} {status}")
    logger.info("-" * 60)

    if not all(results.values()):
        logger.error("One or more downloads failed. Check network and retry.")
        sys.exit(1)

    logger.info("All datasets ready.")
    logger.info("Run ingestion next:")
    logger.info("uv run python pipeline/ingestion.py --corpus all")


if __name__ == "__main__":
    main()
