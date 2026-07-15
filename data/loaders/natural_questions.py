"""
RAGScope — Natural Questions Loader
=====================================
Loads the Natural Questions (NQ) dataset for single-hop factual QA evaluation.

Dataset
-------
* Source  : https://ai.google.com/research/NaturalQuestions
* Licence : CC BY-SA 3.0
* HF hub  : ``natural_questions`` (split ``validation``)
* Paper   : Kwiatkowski et al. (2019), TACL 7:452–466

Usage in this research
----------------------
* 40 development-set queries with Wikipedia ground-truth answers.
* Wikipedia passages are extracted from the long-answer context and
  added to the retrieval corpus, ensuring ground-truth answers are
  retrievable — enabling precise RAGAS answer correctness computation.
"""

from __future__ import annotations

import random
from pathlib import Path

from datasets import load_dataset
from loguru import logger

from config.settings import settings
from data.preprocessing.cleaner import clean, is_meaningful

# ── Constants ─────────────────────────────────────────────────────────────────
DATASET_NAME = "google-research-datasets/natural_questions"
DATASET_CONFIG = "dev"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "natural_questions"
QUERY_SAMPLE_SIZE = 40


# ── Public API ────────────────────────────────────────────────────────────────


def load_queries_and_passages(
    sample_size: int = QUERY_SAMPLE_SIZE,
    seed: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Load NQ development queries and their associated supporting passages.

    Returns a tuple of:
        * ``queries``  — list of query dicts for the evaluation benchmark.
        * ``passages`` — list of passage dicts to add to the retrieval corpus,
                         ensuring the ground-truth answer is findable.

    Query schema::

        {
            "id":           str,
            "query":        str,
            "short_answer": str,           # first short answer token string
            "long_answer":  str,           # cleaned long-answer passage text
            "query_type":   "single_hop",
            "dataset":      "natural_questions",
        }

    Passage schema::

        {
            "id":      str,
            "text":    str,
            "title":   str,
            "dataset": "natural_questions",
        }

    Parameters
    ----------
    sample_size : Number of queries to return. Default 40.
    seed        : Random seed.
    """
    seed = seed if seed is not None else settings.experiment_random_seed
    logger.info(f"Loading Natural Questions (sample_size={sample_size}, seed={seed}) …")

    ds = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="validation",
        cache_dir=str(CACHE_DIR)
    )

    queries: list[dict] = []
    passages: list[dict] = []
    seen_passage_ids: set[str] = set()

    for example in ds:
        question = clean(example.get("question", {}).get("text", ""))
        if not question:
            continue

        # Extract the first short answer (token string representation)
        short_answer = ""
        annotations = example.get("annotations", {})
        for sa_list in annotations.get("short_answers", []):
            if "text" in sa_list and len(sa_list.get("text", [])) > 0:
                short_answer = clean(sa_list["text"][0])
                break

        if not short_answer:
            continue  # Skip examples with no extractable short answer

        # Extract the long-answer passage text from the document tokens
        long_answer_text = _extract_long_answer(example)
        if not long_answer_text or not is_meaningful(long_answer_text):
            continue

        doc_title = example.get("document", {}).get("title", "")
        example_id = str(example["id"])
        passage_id = f"nq_{example_id}"

        queries.append(
            {
                "id": example_id,
                "query": question,
                "short_answer": short_answer,
                "long_answer": long_answer_text,
                "answers": [short_answer],
                "query_type": "single_hop",
                "dataset": "natural_questions",
            }
        )

        if passage_id not in seen_passage_ids:
            seen_passage_ids.add(passage_id)
            passages.append(
                {
                    "id": passage_id,
                    "text": long_answer_text,
                    "title": clean(doc_title),
                    "dataset": "natural_questions",
                }
            )

    # Stratified sample — take up to sample_size queries
    rng = random.Random(seed)
    if len(queries) > sample_size:
        # Sample queries; include their corresponding passages
        sampled_queries = rng.sample(queries, sample_size)
        sampled_ids = {q["id"] for q in sampled_queries}
        sampled_passages = [
            p
            for p in passages
            if p["id"] == f"nq_{list(sampled_ids)[0]}"
            or any(q["id"] in p["id"] for q in sampled_queries)
        ]
    else:
        sampled_queries = queries
        sampled_passages = passages

    logger.info(
        f"Natural Questions loaded: {len(sampled_queries)} queries, "
        f"{len(sampled_passages)} supporting passages."
    )
    return sampled_queries, sampled_passages


def _extract_long_answer(example: dict) -> str:
    """
    Extract the first valid long-answer passage text from an NQ example.

    NQ stores document content as a flat token list. The long-answer
    annotation provides start/end byte positions into that token list.
    We reconstruct the passage text from the token strings.
    """
    annotations = example.get("annotations", {})
    long_answer_candidates = annotations.get("long_answer", [{}])

    doc_tokens = example.get("document", {}).get("tokens", {})
    token_texts = doc_tokens.get("token", [])
    is_html = doc_tokens.get("is_html", [])

    if not token_texts:
        return ""

    for la in long_answer_candidates:
        start_idx = la.get("start_token", -1)
        end_idx = la.get("end_token", -1)

        if start_idx < 0 or end_idx < 0:
            continue

        # Filter out HTML tokens (markup rather than content)
        passage_tokens = [
            t
            for t, h in zip(
                token_texts[start_idx:end_idx],
                is_html[start_idx:end_idx],
            )
            if not h
        ]

        text = clean(" ".join(passage_tokens))
        if is_meaningful(text):
            return text

    return ""
