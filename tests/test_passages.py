"""
Retrieved-passage presentation: citations, sources and score scales.
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")


class TestCitedPassages:
    @pytest.mark.parametrize("answer,expected", [
        ("According to Passage 3, empagliflozin inhibits SGLT2.", {3}),
        ("Passages 1 and 4 agree; passage 2 does not.", {1, 2, 4}),
        ("See passages 2, 3 & 5.", {2, 3, 5}),
        ("Passages 1-3 describe it.", {1, 2, 3}),
        ("Not found in provided context.", set()),
        (None, set()),
    ])
    def test_finds_the_numbers_the_answer_cites(self, answer, expected):
        from dashboard.components.passages import cited_passages

        assert cited_passages(answer) == expected


class TestAnswerHtml:
    def test_tags_citations_and_escapes_html(self):
        from dashboard.components.passages import answer_html

        out = answer_html("Per Passage 3, <b>x</b> & y")
        assert "Passage 3</span>" in out
        assert "&lt;b&gt;" in out and "<b>" not in out


class TestSourceLine:
    def test_bioasq_chunk_links_to_its_pubmed_abstract(self):
        from dashboard.components.passages import source_line

        short, line = source_line("10405209::chunk_0", {"dataset": "bioasq"})
        assert short == "PubMed 10405209"
        assert "https://pubmed.ncbi.nlm.nih.gov/10405209/" in line

    def test_later_chunk_of_an_abstract_says_which_part(self):
        from dashboard.components.passages import source_line

        assert source_line("10405209::chunk_1", {"dataset": "bioasq"})[0] == "PubMed 10405209, part 2"

    def test_own_document_names_file_and_page(self):
        from dashboard.components.passages import source_line

        short, _ = source_line("doc-ab12-p7::chunk_0",
                               {"source": "examples/PMC9849317.pdf", "page": 7})
        assert short == "PMC9849317.pdf, page 7"

    def test_older_document_record_without_metadata_still_gives_the_page(self):
        from dashboard.components.passages import source_line

        assert source_line("doc-ab12cd34ef56ab78-p7::chunk_0", {})[0] == "Your document, page 7"
