"""
Tests for the interactive retrieval-path graph component.

Uses a hand-built ``graph_path`` so the figure logic is tested independently
of the retriever.
"""

from __future__ import annotations

import pytest

pytest.importorskip("plotly")


@pytest.fixture
def graph_path():
    nodes = [
        {"id": "__query__", "label": "What does aspirin treat?", "kind": "query",
         "type": "Query", "layer": 0, "ppr": 0.0, "on_path": True},
        {"id": "Drug_aspirin", "label": "aspirin", "kind": "seed", "type": "ChemicalOrDrug",
         "layer": 1, "ppr": 0.42, "on_path": True, "matcher": "exact",
         "match_score": 1.0, "n_chunks": 1},
        {"id": "Dis_headache", "label": "headache", "kind": "entity", "type": "Disease",
         "layer": 2, "ppr": 0.21, "on_path": True, "n_chunks": 2},
        {"id": "DOC::doc1::chunk_0", "label": "doc1::chunk_0", "kind": "chunk",
         "type": "Retrieved chunk", "layer": 2, "ppr": 0.18, "on_path": True,
         "chunk_id": "doc1::chunk_0", "result_rank": 1, "graph_rank": 1,
         "graph_score": 0.31, "rrf_score": 0.00656, "from_graph_leg": True},
        {"id": "DOC::doc2::chunk_0", "label": "doc2::chunk_0", "kind": "chunk",
         "type": "Retrieved chunk", "layer": 3, "ppr": 0.0, "on_path": True,
         "chunk_id": "doc2::chunk_0", "result_rank": 2, "graph_rank": 2,
         "graph_score": 0.12, "rrf_score": 0.00645, "from_graph_leg": True},
        {"id": "Drug_ibuprofen", "label": "ibuprofen", "kind": "entity",
         "type": "ChemicalOrDrug", "layer": 3, "ppr": 0.09, "on_path": False, "n_chunks": 1},
    ]
    edge = lambda s, t, lbl, on, same=False: {  # noqa: E731
        "source": s, "target": t, "label": lbl, "relation": f"{s} —{lbl}→ {t}",
        "on_path": on, "same_layer": same,
    }
    edges = [
        edge("__query__", "Drug_aspirin", "SEED", True),
        edge("Drug_aspirin", "DOC::doc1::chunk_0", "MENTIONS_COMPOUND", True),
        edge("Drug_aspirin", "Dis_headache", "TREATS", True),
        edge("Dis_headache", "DOC::doc2::chunk_0", "EXTRACTED_FROM", True),
        edge("Dis_headache", "Drug_ibuprofen", "TREATS", False),
    ]
    paths = [
        {"chunk_id": "doc1::chunk_0", "result_rank": 1, "graph_rank": 1,
         "nodes": ["Drug_aspirin", "DOC::doc1::chunk_0"],
         "labels": ["aspirin", "doc1::chunk_0"], "hops": 1, "graph_score": 0.31,
         "contributor": "aspirin", "contributor_ppr": 0.42, "from_graph_leg": True},
        {"chunk_id": "doc2::chunk_0", "result_rank": 2, "graph_rank": 2,
         "nodes": ["Drug_aspirin", "Dis_headache", "DOC::doc2::chunk_0"],
         "labels": ["aspirin", "headache", "doc2::chunk_0"], "hops": 2,
         "graph_score": 0.12, "contributor": "headache", "contributor_ppr": 0.21,
         "from_graph_leg": True},
    ]
    return {"query": "What does aspirin treat?", "nodes": nodes, "edges": edges,
            "paths": paths, "probability_note": "Node probability is PPR mass."}


class TestRetrievalGraphFigure:
    def test_legend_is_a_colour_key(self, graph_path):
        """One trace per colour. Grouping seeds into their own trace made the
        legend swatch show one seed's colour for seeds of every type."""
        from dashboard.components.graph_view import retrieval_graph_figure

        fig = retrieval_graph_figure(graph_path)
        names = {t.name for t in fig.data}
        assert {"Query", "Chemical / drug", "Disease", "Retrieved chunk", "Context links"} <= names
        for t in fig.data:
            if t.mode and "markers" in t.mode and t.name not in ("Relations",):
                colours = t.marker.color
                if isinstance(colours, (list, tuple)):
                    assert len(set(colours)) == 1, f"trace {t.name!r} mixes colours"

    def test_seeds_are_marked_by_outline_not_by_trace(self, graph_path):
        from dashboard.components.graph_view import retrieval_graph_figure

        fig = retrieval_graph_figure(graph_path)
        chem = next(t for t in fig.data if t.name == "Chemical / drug")
        widths = dict(zip(chem.customdata, chem.marker.line.width, strict=True))
        assert widths["Drug_aspirin"] > widths["Drug_ibuprofen"]

    def test_seed_halo_is_theme_neutral_and_not_clickable(self, graph_path):
        """A dark outline vanished on the dark theme; the halo is mid-grey so it
        reads on both. It carries no ids, so clicks select the node beneath."""
        from dashboard.components.graph_view import retrieval_graph_figure

        fig = retrieval_graph_figure(graph_path)
        halo = next(t for t in fig.data if t.name == "Seed entity")
        assert halo.marker.symbol == "circle-open"
        assert halo.customdata is None
        assert len(halo.x) == 1  # one seed in the fixture
        # drawn beneath the entity nodes so it never intercepts a click
        names = [t.name for t in fig.data]
        assert names.index("Seed entity") < names.index("Chemical / drug")

    def test_one_arrow_per_path_edge(self, graph_path):
        from dashboard.components.graph_view import retrieval_graph_figure

        fig = retrieval_graph_figure(graph_path)
        arrows = [a for a in fig.layout.annotations if a.showarrow]
        assert len(arrows) == sum(1 for e in graph_path["edges"] if e["on_path"])

    def test_arrows_point_left_to_right(self, graph_path):
        """Every arrow must end in a later column, which is what makes the
        drawing read as a DAG."""
        from dashboard.components.graph_view import retrieval_graph_figure

        fig = retrieval_graph_figure(graph_path)
        for a in fig.layout.annotations:
            if a.showarrow:
                assert a.x > a.ax

    def test_hiding_context_removes_off_path_nodes(self, graph_path):
        from dashboard.components.graph_view import retrieval_graph_figure

        fig = retrieval_graph_figure(graph_path, show_context=False)
        assert "Context links" not in {t.name for t in fig.data}
        ids = {cd for t in fig.data if t.customdata is not None for cd in t.customdata}
        assert "Drug_ibuprofen" not in ids
        assert "Drug_aspirin" in ids

    def test_highlight_emphasises_one_path(self, graph_path):
        from dashboard.components.graph_view import HIGHLIGHT_EDGE, retrieval_graph_figure

        fig = retrieval_graph_figure(graph_path, highlight_chunk="doc2::chunk_0")
        hi = [a for a in fig.layout.annotations if a.showarrow and a.arrowcolor == HIGHLIGHT_EDGE]
        # query -> aspirin -> headache -> doc2
        assert len(hi) == 3

    def test_chunk_hover_shows_scores_and_text(self, graph_path):
        from dashboard.components.graph_view import retrieval_graph_figure

        fig = retrieval_graph_figure(graph_path, chunk_texts={"doc1::chunk_0": "Aspirin relieves pain."})
        chunk_trace = next(t for t in fig.data if t.name == "Retrieved chunk")
        hover = " ".join(chunk_trace.hovertext)
        assert "Graph score" in hover and "PageRank probability" in hover
        assert "Aspirin relieves pain." in hover

    def test_nodes_carry_their_id_for_click_selection(self, graph_path):
        from dashboard.components.graph_view import retrieval_graph_figure

        fig = retrieval_graph_figure(graph_path)
        chunks = next(t for t in fig.data if t.name == "Retrieved chunk")
        assert set(chunks.customdata) == {"DOC::doc1::chunk_0", "DOC::doc2::chunk_0"}

    def test_chunk_labels_carry_result_rank(self, graph_path):
        from dashboard.components.graph_view import retrieval_graph_figure

        fig = retrieval_graph_figure(graph_path)
        chunks = next(t for t in fig.data if t.name == "Retrieved chunk")
        assert "#1 · doc1::chunk_0" in chunks.text

    def test_query_is_the_title_not_a_clipped_label(self, graph_path):
        from dashboard.components.graph_view import retrieval_graph_figure

        fig = retrieval_graph_figure(graph_path)
        assert "What does aspirin treat?" in fig.layout.title.text


class TestEdgeLabels:
    def test_reversed_traversal_reads_in_arrow_direction(self):
        """Walking entity -> document against a stored document -> entity
        MENTIONS edge must read 'mentioned in', not 'mentions'."""
        from dashboard.components.graph_view import _short_label

        assert _short_label({"label": "MENTIONS_GENE", "reversed": True}) == "mentioned in"
        assert _short_label({"label": "MENTIONS_GENE", "reversed": False}) == "mentions"
        assert _short_label({"label": "INHIBITS", "reversed": True}) == "inhibited by"
        assert _short_label({"label": "TREATS"}) == "treats"


class TestPathTable:
    def test_one_row_per_path_in_rank_order(self, graph_path):
        from dashboard.components.graph_view import path_table_rows

        rows = path_table_rows(graph_path)
        assert [r["Rank"] for r in rows] == [1, 2]
        assert rows[1]["Retrieval path"] == "Query → aspirin → headache → doc2::chunk_0"
        assert rows[1]["Final-hop probability"] == "21.00%"


class TestStreamlitRendering:
    """
    Executes render_retrieval_graph inside Streamlit's own test harness, so the
    widget wiring (toggle, selectbox, plotly_chart with click selection, path
    table) is exercised for real rather than only the figure it draws.
    """

    @staticmethod
    def _app(graph_path):
        def script(gp):
            from dashboard.components.graph_view import render_retrieval_graph

            render_retrieval_graph(gp, chunk_texts={"doc1::chunk_0": "text"}, key="t")

        from streamlit.testing.v1 import AppTest

        return AppTest.from_function(script, args=(graph_path,), default_timeout=20)

    def test_renders_without_exceptions(self, graph_path):
        at = self._app(graph_path).run()
        assert not at.exception, at.exception
        assert len(at.dataframe) == 1  # the path table
        assert at.toggle[0].value is True  # context shown by default

    def test_highlight_selector_lists_every_path(self, graph_path):
        at = self._app(graph_path).run()
        options = at.selectbox[0].options
        assert options[0] == "All paths"
        assert len(options) == 1 + len(graph_path["paths"])

    def test_choosing_a_path_reruns_cleanly(self, graph_path):
        at = self._app(graph_path).run()
        at.selectbox[0].select(at.selectbox[0].options[2]).run()
        assert not at.exception, at.exception
        at.toggle[0].set_value(False).run()
        assert not at.exception, at.exception

    def test_missing_graph_shows_an_explanation(self):
        def script():
            from dashboard.components.graph_view import render_retrieval_graph

            render_retrieval_graph(None, key="t")

        from streamlit.testing.v1 import AppTest

        at = AppTest.from_function(script, default_timeout=20).run()
        assert not at.exception
        assert "no walk to draw" in at.info[0].value


class TestSelectionParsing:
    def test_reads_customdata_from_attribute_and_dict_payloads(self):
        from types import SimpleNamespace

        from dashboard.components.graph_view import _selected_ids

        attr = SimpleNamespace(selection=SimpleNamespace(points=[{"customdata": "Drug_aspirin"}]))
        assert _selected_ids(attr) == ["Drug_aspirin"]
        as_dict = {"selection": {"points": [{"customdata": ["Dis_headache"]}]}}
        assert _selected_ids(as_dict) == ["Dis_headache"]
        assert _selected_ids(None) == []
