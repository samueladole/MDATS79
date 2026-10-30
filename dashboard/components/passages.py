"""
Retrieved passages, presented so a reader can follow the answer's citations.

The generation prompt numbers the retrieved chunks "Passage 1", "Passage 2", …
in rank order, so an answer saying "according to Passage 3" means the third
chunk. This component uses exactly that numbering, marks the passages the
answer cites, names each chunk's source in plain terms (a linked PubMed
abstract, or a file and page), and shows a score only on the scale it really
has: cosine similarity on a 0–1 meter, rank fusion as a rank.

Shared by the Live Monitor (the query just run) and the Query Explorer (a
stored telemetry record).
"""

from __future__ import annotations

import html
import re

import streamlit as st

from dashboard.components.metric_cards import similarity_meter_html

# "Passage 3", "passages 1 and 4", "Passages 2, 3 & 5", "passage 1-2"
_CITE = re.compile(r"\bpassages?\s+((?:\d+\s*(?:,|and|&|or|-|–|to)?\s*)+)", re.IGNORECASE)
_NUM = re.compile(r"\d+")
_RANGE = re.compile(r"(\d+)\s*(?:-|–|to)\s*(\d+)")


def cited_passages(answer: str | None) -> set[int]:
    """Passage numbers the answer refers to, e.g. {1, 3}."""
    found: set[int] = set()
    for m in _CITE.finditer(answer or ""):
        group = m.group(1)
        for a, b in _RANGE.findall(group):
            lo, hi = sorted((int(a), int(b)))
            if hi - lo <= 20:
                found.update(range(lo, hi + 1))
        found.update(int(n) for n in _NUM.findall(group))
    return found


def answer_html(answer: str | None) -> str:
    """The answer, HTML-escaped, with each "Passage N" mention styled as a tag."""
    escaped = html.escape(answer or "")
    tag = ("<span style='background:rgba(42,120,214,0.14);color:inherit;"
           "border-radius:4px;padding:0 4px;font-weight:600;white-space:nowrap'>{}</span>")
    return re.sub(r"\b[Pp]assages?\s+\d+(?:\s*(?:,|and|&|-|–)\s*\d+)*",
                  lambda m: tag.format(m.group(0)), escaped)


def _doc_and_part(chunk_id: str) -> tuple[str, int | None]:
    """'10405209::chunk_0' -> ('10405209', 1)."""
    doc, _, tail = chunk_id.partition("::chunk_")
    return doc, (int(tail) + 1 if tail.isdigit() else None)


def source_line(chunk_id: str, meta: dict) -> tuple[str, str]:
    """(short label for the header, markdown line with a link where possible)."""
    if meta.get("source"):
        page = f", page {meta['page']}" if meta.get("page") else ""
        name = str(meta["source"]).split("/")[-1]
        return f"{name}{page}", f"**Source:** your document `{meta['source']}`{page}"
    doc, part = _doc_and_part(chunk_id)
    part_txt = f", part {part}" if part and part > 1 else ""
    if doc.isdigit() and (meta.get("dataset") in (None, "", "bioasq")):
        url = f"https://pubmed.ncbi.nlm.nih.gov/{doc}/"
        return (f"PubMed {doc}{part_txt}",
                f"**Source:** PubMed abstract [{doc}]({url}){part_txt} · BioASQ benchmark corpus")
    if doc.startswith("doc-"):  # your documents, from an older record without metadata
        page = re.search(r"-p(\d+)$", doc)
        where = f"page {page.group(1)}" if page else doc
        return f"Your document, {where}", f"**Source:** one of your documents, {where}"
    return chunk_id, f"**Source:** `{chunk_id}`"


def render_passages(
    passages: list[dict],
    *,
    answer: str | None,
    strategy: str,
    excerpts: bool = False,
    graph_details=None,
) -> None:
    """Draw the retrieved passages.

    ``passages`` are dicts with ``chunk_id``, ``text``, ``score``, and
    optionally ``score_kind``, ``metadata`` and ``graph``. ``excerpts`` notes
    that the text is the stored excerpt, not the full chunk.
    ``graph_details(passage)`` may draw extra graph provenance in a passage.
    """
    cited = cited_passages(answer)
    st.subheader(f"Retrieved Passages ({len(passages)})")
    is_rrf = any((p.get("score_kind") or ("cosine" if strategy == "dense" else "rrf")) == "rrf"
                 for p in passages)
    notes = [
        "Numbered as the model saw them, so **Passage 3** in the answer means "
        "passage 3 below."
    ]
    if cited:
        notes.append(":material/format_quote: marks the passages the answer cites.")
    if is_rrf:
        notes.append(
            f"With {strategy} retrieval, passages are ordered by rank fusion: the "
            "fusion score (about 0.016) only sets the order and is not a similarity, "
            "so the rank is shown instead."
        )
    st.caption(" ".join(notes))

    for i, p in enumerate(passages, 1):
        meta = p.get("metadata") or {"dataset": p.get("dataset")}
        if p.get("source"):
            meta = {**meta, "source": p["source"], "page": p.get("page")}
        short, line = source_line(p.get("chunk_id", ""), meta)
        kind = p.get("score_kind") or ("cosine" if strategy == "dense" else "rrf")
        score = float(p.get("score") or 0.0)
        score_txt = (f"similarity {score:.2f}" if kind == "cosine"
                     else f"rank {i} of {len(passages)}" if kind == "rrf" else "")
        cited_here = i in cited
        label = (f"**Passage {i}**  ·  {short}"
                 + (f"  ·  {score_txt}" if score_txt else "")
                 + ("  ·  cited in the answer" if cited_here else ""))
        with st.expander(label, icon=":material/format_quote:" if cited_here else None):
            st.markdown(line)
            if kind == "cosine":
                st.markdown(similarity_meter_html(score, label="Cosine similarity to the query"),
                            unsafe_allow_html=True)
            elif kind == "rrf":
                st.caption(f"Rank-fusion score {score:.4f} (orders the passages; not a similarity)")
            if graph_details and p.get("graph"):
                graph_details(p)
            st.markdown(p.get("text", ""))
            if excerpts:
                st.caption("Stored excerpt: the telemetry record keeps the first 500 "
                           "characters of each passage.")
