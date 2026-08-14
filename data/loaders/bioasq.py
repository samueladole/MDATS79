"""
RAGScope — BioASQ Loader
===========================
Loads the BioASQ biomedical QA dataset for use as the retrieval corpus
and a stratified query evaluation set spanning three roles derived from
BioASQ's own challenge structure: Phase A (passage retrieval) and
Phase B (factoid/list and summary/yes-no question answering).

Dataset
-------
* Source  : https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq
* Licence : CC BY 2.5
* HF hub  : ``rag-datasets/rag-mini-bioasq``, configs ``text-corpus`` and
  ``question-answer-passages``
* Derived from the official BioASQ Task 11b training dataset. The full
  official BioASQ release (question-type labels, exact/ideal answers,
  RDF triples, ~23M PubMed abstracts) requires registration at
  bioasq.org and is not used here — this mirror provides a bounded,
  freely downloadable corpus (40,221 PubMed-abstract passages) and
  4,719 question/answer pairs with relevance judgements
  (``relevant_passage_ids``), which is enough to reconstruct a
  dissertation-scale benchmark without a multi-million-document ingest.

Role mapping
------------
This mirror carries no official question-type field (no
factoid/list/summary/yesno label), so the three evaluation roles are
approximated from answer shape and relevance-judgement count. Each
query is assigned to exactly one role by a single deterministic pass
over the full QA pool (see ``_build_role_pools``), so calling any two
of the three public loader functions with the same seed can never
return overlapping queries:

* **Phase A** (passage retrieval): sampled from whatever remains after
  the two roles below have been carved out, stratified by
  ``len(relevant_passage_ids)`` — single-relevant vs. multi-relevant —
  reflecting that retrieval quality, not answer shape, is what this
  role tests.
* **Factoid** (short exact answers): answers of 6 words or fewer,
  approximating BioASQ's convention that "exact answers" are short
  entity/phrase strings.
* **Summary** (multi-document synthesis): the remaining long-answer
  rows, split into yes/no answers (answer text starting with
  "yes"/"no", e.g. "Yes, papilin is a secreted protein") and free-text
  summary answers, reflecting BioASQ's own yes/no and summary question
  types.

Usage in this research
-----------------------
* All 40,221 corpus passages indexed into ChromaDB (already a bounded,
  dissertation-appropriate size — no further subsampling needed).
* 200 queries total (60 Phase A + 40 factoid + 100 summary), a split
  weighted toward the summary role since multi-document synthesis is
  the most demanding condition for faithfulness and the most
  diagnostic for hallucination detection.

The loader uses HuggingFace ``datasets`` for reproducible, cached
downloading. Raw files land in ``data/raw/bioasq/``.
"""

from __future__ import annotations

import ast
import random
from pathlib import Path

from datasets import load_dataset
from loguru import logger

from config.settings import settings
from data.preprocessing.cleaner import clean, is_meaningful

# ── Constants ─────────────────────────────────────────────────────────────────
DATASET_NAME = "rag-datasets/rag-mini-bioasq"
CORPUS_CONFIG = "text-corpus"
QA_CONFIG = "question-answer-passages"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "bioasq"

PHASE_A_COUNT = 60
SINGLE_RELEVANT_COUNT = 30
MULTI_RELEVANT_COUNT = 30
FACTOID_COUNT = 40
SUMMARY_COUNT = 100
YESNO_COUNT = 50
SUMMARY_LONG_COUNT = 50

FACTOID_MAX_WORDS = 6  # answers this short or shorter are treated as "exact answers"


# ── Public API ────────────────────────────────────────────────────────────────


def load_corpus(seed: int | None = None) -> list[dict]:
    """
    Load the full BioASQ passage corpus as the retrieval corpus.

    Each returned dict has the schema::

        {
            "id":      str,   # unique passage identifier (source PubMed ID)
            "text":    str,   # cleaned passage text
            "title":   str,   # always "" — the source has no separate title field
            "dataset": "bioasq",
        }

    Parameters
    ----------
    seed : Unused (kept for interface parity with the other loaders —
           the full corpus is always returned, since 40,221 passages is
           already a bounded, dissertation-appropriate size).
    """
    logger.info("Loading BioASQ corpus (full text-corpus config) …")

    ds = load_dataset(DATASET_NAME, name=CORPUS_CONFIG, cache_dir=str(CACHE_DIR))
    split = ds["passages"]

    pool: list[dict] = []
    for example in split:
        cleaned = clean(example.get("passage", ""))
        if not is_meaningful(cleaned):
            continue
        pool.append(
            {
                "id": str(example["id"]),
                "text": cleaned,
                "title": "",
                "dataset": "bioasq",
            }
        )

    logger.info(f"BioASQ corpus loaded: {len(pool):,} passages.")
    return pool


def load_phase_a_queries(sample_size: int = PHASE_A_COUNT, seed: int | None = None) -> list[dict]:
    """
    Load the Phase A (passage-retrieval) role queries.

    Each returned dict has the schema::

        {
            "id":           str,
            "query":        str,
            "answers":      list[str],
            "query_type":   "single_relevant" | "multi_relevant",
            "relevant_passage_ids": list[str],
            "dataset":      "bioasq_phase_a",
        }
    """
    seed = seed if seed is not None else settings.experiment_random_seed
    pools = _build_role_pools(seed=seed)
    single = pools["phase_a_single"]
    multi = pools["phase_a_multi"]

    n_single = min(sample_size // 2, len(single))
    n_multi = min(sample_size - n_single, len(multi))

    result = single[:n_single] + multi[:n_multi]
    random.Random(seed).shuffle(result)

    logger.info(f"BioASQ Phase A queries loaded: {n_single} single-relevant, {n_multi} multi-relevant.")
    return result


def load_factoid_queries(sample_size: int = FACTOID_COUNT, seed: int | None = None) -> list[dict]:
    """
    Load the factoid/list role queries (short exact answers).

    Each returned dict has the schema::

        {
            "id":           str,
            "query":        str,
            "answers":      list[str],
            "query_type":   "factoid",
            "relevant_passage_ids": list[str],
            "dataset":      "bioasq_factoid",
        }
    """
    seed = seed if seed is not None else settings.experiment_random_seed
    pools = _build_role_pools(seed=seed)
    pool = pools["factoid"]

    sampled = pool[: min(sample_size, len(pool))]
    logger.info(f"BioASQ factoid queries loaded: {len(sampled)}.")
    return sampled


def load_summary_queries(sample_size: int = SUMMARY_COUNT, seed: int | None = None) -> list[dict]:
    """
    Load the summary/yes-no role queries (multi-document synthesis).

    Each returned dict has the schema::

        {
            "id":           str,
            "query":        str,
            "answers":      list[str],
            "query_type":   "yesno" | "summary",
            "relevant_passage_ids": list[str],
            "dataset":      "bioasq_summary",
        }
    """
    seed = seed if seed is not None else settings.experiment_random_seed
    pools = _build_role_pools(seed=seed)
    yesno = pools["summary_yesno"]
    summary = pools["summary_long"]

    n_yesno = min(sample_size // 2, len(yesno))
    n_summary = min(sample_size - n_yesno, len(summary))

    result = yesno[:n_yesno] + summary[:n_summary]
    random.Random(seed).shuffle(result)

    logger.info(f"BioASQ summary queries loaded: {n_yesno} yes/no, {n_summary} summary.")
    return result


# ── Internal helpers ─────────────────────────────────────────────────────────


def _build_role_pools(seed: int) -> dict[str, list[dict]]:
    """
    Partition the full BioASQ QA pool into disjoint, pre-sampled role
    buckets in a single deterministic pass, so independent calls to the
    three public loader functions above (with the same seed) never draw
    overlapping queries — critically, the Phase A pool is built from
    whatever remains *after* the summary role's rows have already been
    selected, not from the same shared candidate pool (an earlier draft
    of this function sampled Phase A and summary independently from an
    overlapping pool, which could hand both roles the same question).

    Classification priority per row: factoid (short answer), then
    yes/no (answer starts with "yes"/"no"), then everything else is a
    long-answer candidate shared by the summary and Phase A roles —
    summary claims its share first, Phase A draws only from the
    remainder. Not cached: re-run on every call, matching the pattern
    of the other loaders (cheap — HuggingFace's own cache makes the
    underlying ``load_dataset`` call fast after the first download).
    """
    ds = load_dataset(DATASET_NAME, name=QA_CONFIG, cache_dir=str(CACHE_DIR))
    split = ds["test"]

    factoid_pool: list[dict] = []
    yesno_pool: list[dict] = []
    long_answer_pool: list[dict] = []

    for example in split:
        question = clean(example.get("question", ""))
        answer = clean(example.get("answer", ""))
        if not question or not answer:
            continue

        try:
            relevant_ids = [str(pid) for pid in ast.literal_eval(example["relevant_passage_ids"])]
        except (ValueError, SyntaxError):
            continue
        if not relevant_ids:
            continue

        word_count = len(answer.split())
        answer_lower = answer.lower()
        record_base = {
            "id": str(example["id"]),
            "query": question,
            "answers": [answer],
            "relevant_passage_ids": relevant_ids,
        }

        if word_count <= FACTOID_MAX_WORDS:
            factoid_pool.append({**record_base, "query_type": "factoid", "dataset": "bioasq_factoid"})
        elif answer_lower.startswith("yes") or answer_lower.startswith("no"):
            yesno_pool.append({**record_base, "query_type": "yesno", "dataset": "bioasq_summary"})
        else:
            long_answer_pool.append(record_base)

    rng = random.Random(seed)
    rng.shuffle(factoid_pool)
    rng.shuffle(yesno_pool)
    rng.shuffle(long_answer_pool)

    # Summary claims its share of the long-answer pool first …
    n_summary_long = min(SUMMARY_LONG_COUNT, len(long_answer_pool))
    summary_long_pool = [
        {**r, "query_type": "summary", "dataset": "bioasq_summary"} for r in long_answer_pool[:n_summary_long]
    ]
    # … and Phase A draws only from what's left, so the two roles can
    # never both sample the same question.
    phase_a_candidates = long_answer_pool[n_summary_long:]
    phase_a_single_pool = [
        {**r, "query_type": "single_relevant", "dataset": "bioasq_phase_a"}
        for r in phase_a_candidates
        if len(r["relevant_passage_ids"]) == 1
    ]
    phase_a_multi_pool = [
        {**r, "query_type": "multi_relevant", "dataset": "bioasq_phase_a"}
        for r in phase_a_candidates
        if len(r["relevant_passage_ids"]) > 1
    ]

    logger.debug(
        f"BioASQ role pools built: factoid={len(factoid_pool)}, yesno={len(yesno_pool)}, "
        f"summary_long={len(summary_long_pool)}, phase_a_single={len(phase_a_single_pool)}, "
        f"phase_a_multi={len(phase_a_multi_pool)}."
    )

    return {
        "factoid": factoid_pool,
        "summary_yesno": yesno_pool,
        "summary_long": summary_long_pool,
        "phase_a_single": phase_a_single_pool,
        "phase_a_multi": phase_a_multi_pool,
    }
