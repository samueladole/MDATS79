"""
Run Query button behaviour on the Live Monitor page.

Driven through Streamlit's own AppTest harness with the RAG pipeline replaced
by a stub, so no LLM, vector store or RAGAS evaluation is involved.

The button must be disabled for the whole time the pipeline is running — on
the first query as well as later ones — and must come back on every exit path:
success, pipeline error, and an empty submission.
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

PAGE = "dashboard/pages/01_live_monitor.py"
BUTTON = "run_query_btn"


def _result(query: str):
    from pipeline.generation.llm_client import GenerationResponse
    from pipeline.rag import RAGResult
    from telemetry.token_counter import TokenUsage

    return RAGResult(
        query=query,
        answer="stub answer",
        retrieved_chunks=[],
        retrieval_telemetry={"strategy": "dense", "retrieval_ms": 1.0},
        generation_response=GenerationResponse(
            text="stub answer", model="stub", prompt_tokens=1,
            completion_tokens=1, total_tokens=2, generation_ms=1.0,
        ),
        token_usage=TokenUsage(
            prompt_tokens=1, completion_tokens=1, total_tokens=2, estimated_cost_usd=0.0
        ),
        e2e_ms=2.0,
    )


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Replace RAGPipeline. ``mode`` selects how the stub behaves:

    - "ok":           returns a result
    - "error":        answer() raises, as a failed Ollama call would
    - "stop":         answer() calls st.stop() — freezes the page mid-run, so
                      the test can inspect what was rendered while it was busy
    - "stop_in_eval": freezes the page during evaluation instead
    """
    calls = {"n": 0, "mode": "ok"}

    class StubPipeline:
        def __init__(self, *args, **kwargs):
            pass

        def answer(self, query_text, **kwargs):
            import streamlit as st

            calls["n"] += 1
            if calls["mode"] == "stop":
                st.stop()
            if calls["mode"] == "error":
                raise RuntimeError("ollama unreachable")
            from types import SimpleNamespace

            return SimpleNamespace(query=query_text, answer="stub answer")

        def evaluate(self, draft, **kwargs):
            import streamlit as st

            if calls["mode"] == "stop_in_eval":
                st.stop()
            return _result(draft.query)

    import pipeline.rag

    monkeypatch.setattr(pipeline.rag, "RAGPipeline", StubPipeline)
    return calls


def _app():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(PAGE, default_timeout=30).run()


def _submit(at, text: str = "What does aspirin treat?"):
    at.main.text_area[0].input(text)
    return at.button(key=BUTTON).click().run()


class TestRunButton:
    def test_disabled_while_the_first_query_runs(self, stub_pipeline):
        """The first query must be protected too, not only later ones."""
        stub_pipeline["mode"] = "stop"
        at = _submit(_app())
        assert stub_pipeline["n"] == 1
        assert at.button(key=BUTTON).disabled is True

    def test_disabled_while_a_later_query_runs(self, stub_pipeline):
        at = _submit(_app())
        stub_pipeline["mode"] = "stop"
        at.button(key=BUTTON).click().run()
        assert stub_pipeline["n"] == 2
        assert at.button(key=BUTTON).disabled is True

    def test_re_enabled_after_every_query(self, stub_pipeline):
        """Regression: an earlier version re-derived the flag from whether a
        previous result existed, so after the first query the button disabled
        itself and was never re-enabled for the rest of the session."""
        at = _app()
        at.main.text_area[0].input("What does aspirin treat?")
        for i in range(1, 4):
            at.button(key=BUTTON).click().run()
            assert not at.exception, at.exception
            assert stub_pipeline["n"] == i, f"query {i} did not run"
            assert at.button(key=BUTTON).disabled is False, f"locked after query {i}"

    def test_error_re_enables_and_is_reported(self, stub_pipeline):
        stub_pipeline["mode"] = "error"
        at = _submit(_app())
        assert at.button(key=BUTTON).disabled is False
        assert any("ollama unreachable" in e.value for e in at.error)

    def test_recovers_after_an_error(self, stub_pipeline):
        stub_pipeline["mode"] = "error"
        at = _submit(_app())
        stub_pipeline["mode"] = "ok"
        at.button(key=BUTTON).click().run()
        assert stub_pipeline["n"] == 2
        assert at.button(key=BUTTON).disabled is False

    def test_empty_query_neither_runs_nor_locks(self, stub_pipeline):
        at = _app()
        at.button(key=BUTTON).click().run()
        assert stub_pipeline["n"] == 0
        assert at.button(key=BUTTON).disabled is False
        assert any("enter a query" in w.value for w in at.warning)


class TestAnswerBeforeEvaluation:
    def test_answer_is_shown_while_evaluation_runs(self, stub_pipeline):
        """Evaluation is most of the wait; the answer must not wait for it."""
        stub_pipeline["mode"] = "stop_in_eval"
        at = _submit(_app())
        assert any("stub answer" in m.value for m in at.markdown)
        assert at.button(key=BUTTON).disabled is True
