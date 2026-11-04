"""
The LightRAG comparison's question files and hit scoring.

LightRAG itself is not a project dependency, so only the parts that do not
import it are tested here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "graph"))

import lightrag_compare as lc  # noqa: E402


def test_questions_file_with_and_without_expected_sources(tmp_path):
    f = tmp_path / "q.txt"
    f.write_text(
        "# my questions\n"
        "\n"
        "What does the trial conclude?\n"
        "Does empagliflozin reduce tissue sodium? | trial.pdf, page 3\n"
        "Which genes were studied? | notes.md\n"
        "Dose? | trial.pdf p 2\n"
    )
    qs = lc.read_questions(f)
    assert [q["question"] for q in qs] == [
        "What does the trial conclude?", "Does empagliflozin reduce tissue sodium?",
        "Which genes were studied?", "Dose?"]
    assert qs[0]["expected"] is None
    assert qs[1]["expected"] == {"file": "trial.pdf", "page": 3}
    assert qs[2]["expected"] == {"file": "notes.md", "page": None}
    assert qs[3]["expected"] == {"file": "trial.pdf", "page": 2}


def test_hit_needs_the_file_and_the_page_when_given():
    sources = {"a::chunk_0": ("papers/trial.pdf", 3), "b::chunk_0": ("notes.md", None)}
    assert lc._is_hit(["a::chunk_0"], {"file": "trial.pdf", "page": 3}, sources)
    assert not lc._is_hit(["a::chunk_0"], {"file": "trial.pdf", "page": 4}, sources)
    assert lc._is_hit(["b::chunk_0", "x"], {"file": "NOTES.md", "page": None}, sources)
    assert not lc._is_hit([], {"file": "notes.md", "page": None}, sources)


def test_each_knowledge_base_keeps_its_own_results():
    lc.use_knowledge_base("documents")
    docs = lc.RESULTS
    lc.use_knowledge_base("bioasq")
    assert docs != lc.RESULTS
    assert docs.parent == lc.RESULTS.parent / "documents"
