"""
RAGScope — MS MARCO Loader
============================
Loads the MS MARCO passage ranking dataset for use as the primary
retrieval corpus and a stratified query evaluation set.

Dataset
-------
* Source  : https://microsoft.github.io/msmarco/
* Licence : MIT
* HF hub  : ``ms_marco`` (config ``v2.1``)

Usage in this research
----------------------
* 50,000 passages from the passage corpus indexed into ChromaDB.
* 60 development-set queries (30 single-answer, 30 multi-passage)
  form part of the 200-query evaluation benchmark.

The loader uses HuggingFace ``datasets`` for reproducible, cached
downloading. Raw files land in ``data/raw/msmarco/``.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path

from datasets import load_dataset
from loguru import logger

from config.settings import settings
from data.preprocessing.cleaner import clean, is_meaningful

# ── Constants ─────────────────────────────────────────────────────────────────
DATASET_NAME = "microsoft/ms_marco"
DATASET_CONFIG = "v2.1"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "msmarco"

CORPUS_SAMPLE_SIZE = 50_000  # passages indexed into ChromaDB
QUERY_SAMPLE_SIZE = 60  # queries used in the evaluation benchmark
SINGLE_ANSWER_COUNT = 30  # queries with a single relevant passage
MULTI_PASSAGE_COUNT = 30  # queries with multiple relevant passages


# ── Public API ────────────────────────────────────────────────────────────────


def load_corpus(
    sample_size: int = CORPUS_SAMPLE_SIZE,
    seed: int | None = None,
) -> list[dict]:
    """
    Load a random sample of MS MARCO passages as the retrieval corpus.

    Each returned dict has the schema::

        {
            "id":      str,   # unique passage identifier
            "text":    str,   # cleaned passage text
            "title":   str,   # passage title (may be empty)
            "dataset": "msmarco",
        }

    Parameters
    ----------
    sample_size : Number of passages to return. Default 50,000.
    seed        : Random seed for reproducibility. Defaults to ``settings.experiment_random_seed``.

    Returns
    -------
    list[dict]
    """
    seed = seed if seed is not None else settings.experiment_random_seed
    logger.info(f"Loading MS MARCO corpus (sample_size={sample_size}, seed={seed}) …")

    ds = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="train",
        cache_dir=str(CACHE_DIR)
    )

    # MS MARCO v2.1 stores passages in a nested list under "passages".
    # Flatten all passages from all query examples into a deduplicated pool.
    seen_ids: set[str] = set()
    pool: list[dict] = []

    for example in ds:
        for i, passage_text in enumerate(example.get("passages", {}).get("passage_text", [])):
            pid = f"msmarco_{example['query_id']}_{i}"
            if pid in seen_ids:
                continue
            seen_ids.add(pid)

            cleaned = clean(passage_text)
            if not is_meaningful(cleaned):
                continue

            pool.append(
                {
                    "id": pid,
                    "text": cleaned,
                    "title": "",
                    "dataset": "msmarco",
                }
            )

            if len(pool) >= sample_size * 3:
                # Collected enough candidates — stop early.
                break
        if len(pool) >= sample_size * 3:
            break

    rng = random.Random(seed)
    sample = rng.sample(pool, min(sample_size, len(pool)))
    logger.info(f"MS MARCO corpus loaded: {len(sample):,} passages.")
    return sample


def load_queries(
    sample_size: int = QUERY_SAMPLE_SIZE,
    seed: int | None = None,
) -> list[dict]:
    """
    Load a stratified sample of MS MARCO development queries.

    Returns a mix of:
        * Single-answer queries  (one relevant passage in the corpus)
        * Multi-passage queries  (multiple relevant passages)

    Each returned dict has the schema::

        {
            "id":           str,
            "query":        str,
            "answers":      list[str],
            "query_type":   "single_answer" | "multi_passage",
            "dataset":      "msmarco",
        }

    Parameters
    ----------
    sample_size : Total queries to return. Default 60.
    seed        : Random seed.
    """
    seed = seed if seed is not None else settings.experiment_random_seed
    logger.info(f"Loading MS MARCO queries (sample_size={sample_size}, seed={seed}) …")

    ds = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="train",
        cache_dir=str(CACHE_DIR)
    )

    single: list[dict] = []
    multi: list[dict] = []

    for example in ds:
        query = clean(example.get("query", ""))
        if not query:
            continue

        answers = example.get("answers", [])
        n_passages = len(example.get("passages", {}).get("passage_text", []))
        query_type = "multi_passage" if n_passages > 1 else "single_answer"

        record = {
            "id": str(example["query_id"]),
            "query": query,
            "answers": [clean(a) for a in answers if a],
            "query_type": query_type,
            "dataset": "msmarco",
        }

        if len(single) >= SINGLE_ANSWER_COUNT * 3 and len(multi) >= MULTI_PASSAGE_COUNT * 3:
            logger.info(f"Collected enough candidate queries: {len(single)} single-answer, {len(multi)} multi-passage.")
            break
        if query_type == "single_answer" and len(single) < SINGLE_ANSWER_COUNT * 3:
            single.append(record)
        elif query_type == "multi_passage" and len(multi) < MULTI_PASSAGE_COUNT * 3:
            multi.append(record)

    rng = random.Random(seed)
    sampled_single = rng.sample(single, min(SINGLE_ANSWER_COUNT, len(single)))
    sampled_multi = rng.sample(multi, min(MULTI_PASSAGE_COUNT, len(multi)))
    result = sampled_single + sampled_multi
    rng.shuffle(result)

    logger.info(
        f"MS MARCO queries loaded: {len(sampled_single)} single-answer, "
        f"{len(sampled_multi)} multi-passage."
    )
    return result


def stream_corpus() -> Iterator[dict]:
    """
    Stream MS MARCO passages one at a time (memory-efficient alternative
    to ``load_corpus`` for very large ingestion runs).
    """
    ds = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="train",
        cache_dir=str(CACHE_DIR),
        streaming=True
    )
    for example in ds:
        for i, passage_text in enumerate(example.get("passages", {}).get("passage_text", [])):
            cleaned = clean(passage_text)
            if is_meaningful(cleaned):
                yield {
                    "id": f"msmarco_{example['query_id']}_{i}",
                    "text": cleaned,
                    "title": "",
                    "dataset": "msmarco",
                }
