"""Tests for the telemetry layer."""

from __future__ import annotations

import time

from telemetry.timer import Timer
from telemetry.token_counter import TokenCounter, TokenUsage


class TestTimer:
    def test_elapsed_is_positive(self):
        with Timer("test") as t:
            time.sleep(0.01)
        assert t.elapsed_ms > 0

    def test_elapsed_is_roughly_correct(self):
        with Timer() as t:
            time.sleep(0.05)
        assert 40 <= t.elapsed_ms <= 200  # generous bounds for CI

    def test_repr(self):
        with Timer("stage") as t:
            pass
        assert "stage" in repr(t)
        assert "elapsed_ms" in repr(t)

    def test_zero_before_exit(self):
        t = Timer()
        t.__enter__()
        assert t.elapsed_ms == 0.0
        t.__exit__(None, None, None)


class TestTokenCounter:
    def setup_method(self):
        self.counter = TokenCounter()

    def test_build_usage_zero_cost(self):
        usage = self.counter.build_usage(100, 50)
        assert isinstance(usage, TokenUsage)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.estimated_cost_usd == 0.0  # default rate is $0.00

    def test_build_usage_to_dict(self):
        usage = self.counter.build_usage(10, 20)
        d = usage.to_dict()
        assert set(d.keys()) == {
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "estimated_cost_usd",
        }


class TestTelemetryLogger:
    def test_log_and_load(self, tmp_path):
        from pipeline.vectorstore import RetrievedChunk
        from telemetry.logger import TelemetryLogger, build_record
        from telemetry.token_counter import TokenUsage

        logger = TelemetryLogger(store_dir=tmp_path)
        chunk = RetrievedChunk(
            chunk_id="doc1::chunk_0",
            text="test passage",
            score=0.85,
            metadata={"dataset": "bioasq"},
        )
        usage = TokenUsage(
            prompt_tokens=100, completion_tokens=50, total_tokens=150, estimated_cost_usd=0.0
        )
        record = build_record(
            query_id="q1",
            query="What is NLP?",
            dataset="bioasq",
            query_type="single_hop",
            llm_model="llama3",
            retrieval_strategy="dense",
            retrieved_chunks=[chunk],
            retrieval_telemetry={"retrieval_ms": 120.0, "embed_query_ms": 30.0},
            answer="NLP is Natural Language Processing.",
            generation_ms=800.0,
            e2e_ms=920.0,
            token_usage=usage,
        )
        path = logger.log(record)
        assert path.exists()

        loaded = logger.load_all()
        assert len(loaded) == 1
        assert loaded[0]["query"] == "What is NLP?"
        assert loaded[0]["retrieval_strategy"] == "dense"
        # build_record() uses a flat schema (documented in telemetry/logger.py:
        # "intentionally flat to simplify pandas ingestion") — token fields are
        # top-level, not nested under a "token_usage" key.
        assert loaded[0]["prompt_tokens"] == 100
        assert loaded[0]["completion_tokens"] == 50

    def test_count(self, tmp_path):
        from pipeline.vectorstore import RetrievedChunk
        from telemetry.logger import TelemetryLogger, build_record
        from telemetry.token_counter import TokenUsage

        logger = TelemetryLogger(store_dir=tmp_path)
        assert logger.count() == 0
        usage = TokenUsage(10, 5, 15, 0.0)
        chunk = RetrievedChunk("c::0", "text", 0.9, {})
        for i in range(3):
            record = build_record(
                query_id=f"q{i}",
                query=f"Query {i}",
                dataset="test",
                query_type="test",
                llm_model="llama3",
                retrieval_strategy="dense",
                retrieved_chunks=[chunk],
                retrieval_telemetry={"retrieval_ms": 10.0, "embed_query_ms": 5.0},
                answer="Answer",
                generation_ms=100.0,
                e2e_ms=110.0,
                token_usage=usage,
            )
            logger.log(record)
        assert logger.count() == 3
