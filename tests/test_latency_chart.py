"""
Tests for the pipeline flow Sankey (Live Monitor and Query Explorer).

The figure is built without Streamlit, so its structure is tested directly.
"""

from __future__ import annotations

import pytest

pytest.importorskip("plotly")

HYBRID = {"strategy": "hybrid", "embed_query_ms": 8.25, "dense_search_ms": 38.94,
          "bm25_search_ms": 217.33, "rrf_fusion_ms": 0.08}


def _sankey(tel=HYBRID, gen=6782.5, ev=40962.5, e2e=48011.1, ran=True):
    from dashboard.components.latency_chart import pipeline_flow_figure

    return pipeline_flow_figure(tel, gen, ev, e2e, ran).data[0]


def _names(s) -> list[str]:
    return [lbl.split("</b>")[0].removeprefix("<b>") for lbl in s.node.label]


class TestPipelineFlow:
    def test_three_columns_total_phases_steps(self):
        s = _sankey()
        assert _names(s) == [
            "End to end", "Retrieval (hybrid)", "Generation (LLM)", "Evaluation (RAGAS)",
            "Other overhead", "Query embedding", "Dense search", "BM25 search",
            "Rank fusion (RRF)",
        ]
        assert len(set(s.node.x)) == 3  # one x position per column

    def test_other_overhead_is_the_unattributed_remainder(self):
        s = _sankey()
        other = next(lbl for lbl in s.node.label if lbl.startswith("<b>Other overhead"))
        # 48011.1 - (264.6 + 6782.5 + 40962.5) = 1.5 ms
        assert "1.5 ms" in other

    def test_labels_show_exact_times_in_readable_units(self):
        s = _sankey()
        text = " ".join(s.node.label)
        assert "40.96 s" in text and "85.3%" in text   # seconds above 1 s
        assert "217 ms" in text and "0.08 ms" in text  # ms below, exact for tiny steps

    def test_tiny_stages_are_drawn_visible_but_labelled_exactly(self):
        from dashboard.components.latency_chart import MIN_SHARE

        s = _sankey()
        to_other = s.link.value[3]  # End to end -> Other overhead
        assert to_other >= MIN_SHARE * 48011.1 * 0.999
        assert "1.5 ms" in s.node.label[4]

    def test_retrieval_steps_fill_exactly_the_retrieval_band(self):
        s = _sankey()
        into_retrieval = s.link.value[0]
        out_of_retrieval = sum(v for src, v in zip(s.link.source, s.link.value, strict=True)
                               if src == 1)
        assert out_of_retrieval == pytest.approx(into_retrieval)

    def test_evaluation_is_omitted_when_it_did_not_run(self):
        s = _sankey(ev=0.0, e2e=7100.0, ran=False)
        assert "Evaluation (RAGAS)" not in _names(s)

    @pytest.mark.parametrize("strategy,steps", [
        ("dense", ["Query embedding", "Vector search"]),
        ("graph", ["Query embedding", "Dense search", "Entity seeding", "Graph traversal",
                   "Chunk scoring", "Rank fusion (RRF)"]),
    ])
    def test_each_strategy_shows_its_own_retrieval_steps(self, strategy, steps):
        s = _sankey(tel={"strategy": strategy})
        assert _names(s)[5:] == steps

    def test_nodes_are_placed_explicitly_so_labels_cannot_collide(self):
        s = _sankey()
        assert s.arrangement == "fixed"
        step_ys = list(s.node.y[5:])
        assert step_ys == sorted(step_ys) and len(set(step_ys)) == len(step_ys)


class TestFormatDuration:
    @pytest.mark.parametrize("ms,expected", [
        (48011.1, "48.01 s"),
        (1000.0, "1.00 s"),
        (999.4, "999 ms"),
        (217.33, "217 ms"),
        (8.25, "8.2 ms"),
        (0.08, "0.08 ms"),
        (None, "—"),
    ])
    def test_seconds_from_one_second_milliseconds_below(self, ms, expected):
        from dashboard.components.formatting import format_duration

        assert format_duration(ms) == expected
