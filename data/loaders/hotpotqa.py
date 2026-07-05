"""
RAGScope — HotpotQA Loader
============================
Loads the HotpotQA multi-hop question answering dataset.

Dataset
-------
* Source  : https://hotpotqa.github.io/
* Licence : CC BY-SA 4.0
* HF hub  : ``hotpot_qa`` (config ``fullwiki``)
* Paper   : Yang et al. (2018), EMNLP, pp. 2369–2380

Usage in this research
----------------------
* 100 development-set queries from the full-wiki setting:
    - 50 "bridge" questions  (sequential reasoning across two entities)
    - 50 "comparison" questions (attribute comparison between two entities)
* Supporting passages from the ``context`` field are added to the
  retrieval corpus to ensure answers are retrievable.
* HotpotQA is the stress-test component of the benchmark — multi-hop
  reasoning is the hardest condition for RAG faithfulness and the most
  diagnostic test for hallucination detection.
"""

from __future__ import annotations

import random
from pathlib import Path

from datasets import load_dataset
from loguru import logger

from config.settings import settings
from data.preprocessing.cleaner import clean, is_meaningful

# ── Constants ─────────────────────────────────────────────────────────────────
DATASET_NAME = "hotpotqa/hotpot_qa"
DATASET_CONFIG = "fullwiki"
CACHE_DIR = Path("data/raw/hotpotqa")

QUERY_SAMPLE_SIZE = 100
BRIDGE_COUNT = 50
COMPARISON_COUNT = 50


# ── Public API ────────────────────────────────────────────────────────────────


def load_queries_and_passages(
    sample_size: int = QUERY_SAMPLE_SIZE,
    seed: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Load HotpotQA development queries and their supporting passages.

    Returns a tuple of:
        * ``queries``  — list of query dicts for the evaluation benchmark.
        * ``passages`` — list of passage dicts to add to the retrieval corpus.

    Query schema::

        {
            "id":           str,
            "query":        str,
            "answer":       str,
            "answers":      list[str],       # [answer] — unified interface
            "query_type":   "bridge" | "comparison",
            "supporting_facts": list[str],   # titles of supporting documents
            "dataset":      "hotpotqa",
        }

    Passage schema::

        {
            "id":      str,
            "text":    str,
            "title":   str,
            "dataset": "hotpotqa",
        }

    Parameters
    ----------
    sample_size : Total queries to return. Default 100.
    seed        : Random seed.
    """
    seed = seed if seed is not None else settings.experiment_random_seed
    n_bridge = sample_size // 2
    n_comparison = sample_size - n_bridge

    logger.info(
        f"Loading HotpotQA fullwiki (bridge={n_bridge}, comparison={n_comparison}, seed={seed}) …"
    )

    ds = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="validation",
        cache_dir=str(CACHE_DIR)
    )

    bridge_pool: list[dict] = []
    comparison_pool: list[dict] = []
    passages: list[dict] = []
    seen_passage_ids: set[str] = set()

    for example in ds:
        question = clean(example.get("question", ""))
        answer = clean(example.get("answer", ""))
        qtype = example.get("type", "").lower()  # "bridge" or "comparison"

        if not question or not answer:
            continue
        if qtype not in {"bridge", "comparison"}:
            continue

        # Extract supporting passage titles
        sf_titles = list({t for t in example.get("supporting_facts", {}).get("title", [])})

        # Extract context passages (list of [title, sentences])
        context = example.get("context", {})
        ctx_titles = context.get("title", [])
        ctx_sentences = context.get("sentences", [])

        example_passages: list[dict] = []
        for title, sentences in zip(ctx_titles, ctx_sentences):
            passage_text = clean(" ".join(sentences))
            if not is_meaningful(passage_text):
                continue
            pid = f"hotpotqa_{example['id']}_{clean(title)[:40].replace(' ', '_')}"
            if pid not in seen_passage_ids:
                seen_passage_ids.add(pid)
                p = {
                    "id": pid,
                    "text": passage_text,
                    "title": clean(title),
                    "dataset": "hotpotqa",
                }
                passages.append(p)
                example_passages.append(p)

        record = {
            "id": example["id"],
            "query": question,
            "answer": answer,
            "answers": [answer],
            "query_type": qtype,
            "supporting_facts": sf_titles,
            "passage_ids": [p["id"] for p in example_passages],
            "dataset": "hotpotqa",
        }

        if qtype == "bridge" and len(bridge_pool) < n_bridge * 3:
            bridge_pool.append(record)
        elif qtype == "comparison" and len(comparison_pool) < n_comparison * 3:
            comparison_pool.append(record)

        if len(bridge_pool) >= n_bridge * 3 and len(comparison_pool) >= n_comparison * 3:
            break

    rng = random.Random(seed)
    sampled_bridge = rng.sample(bridge_pool, min(n_bridge, len(bridge_pool)))
    sampled_comparison = rng.sample(comparison_pool, min(n_comparison, len(comparison_pool)))
    result = sampled_bridge + sampled_comparison
    rng.shuffle(result)

    logger.info(
        f"HotpotQA loaded: {len(sampled_bridge)} bridge, "
        f"{len(sampled_comparison)} comparison, "
        f"{len(passages)} supporting passages."
    )
    return result, passages


def load_queries(
    sample_size: int = QUERY_SAMPLE_SIZE,
    seed: int | None = None,
) -> list[dict]:
    """Convenience wrapper that returns only queries (no passages)."""
    queries, _ = load_queries_and_passages(sample_size=sample_size, seed=seed)
    return queries
