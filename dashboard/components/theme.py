"""
RAGScope — Shared Dashboard Theme
===================================
Every page (`dashboard/app.py` and `dashboard/pages/*.py`) is a separate
Streamlit script run, so CSS injected in one page does not carry over to
another — this module is imported and called once per page to keep styling
consistent across the whole app.

Icons: Streamlit's own widgets (buttons, expanders, alerts, `st.title`, page
config) natively support Google's Material Symbols via an ``":material/name:"``
shortcode, which is used throughout the dashboard instead of emoji. That
shortcode is resolved by Streamlit's markdown pipeline — it is not guaranteed
to resolve inside a hand-built HTML string passed through
``unsafe_allow_html=True`` (raw HTML blocks aren't reliably re-processed as
markdown), so the small set of icons embedded in custom HTML (the RAGAS
scorecard tiles) use hand-authored inline SVG instead, via ``ICONS`` below.
"""

from __future__ import annotations

import streamlit as st

# ── Inline SVG icons for custom-HTML contexts ─────────────────────────────────
# 18x18, stroke-based, currentColor — inherits whatever `color` the
# surrounding element sets, so one icon works across light/dark themes and
# any label tint without needing per-context colour variants.
ICONS = {
    "monitoring": (
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>'
        "</svg>"
    ),
    "target": (
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>'
        "</svg>"
    ),
    "link": (
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M10 13a5 5 0 0 0 7.07 0l2.83-2.83a5 5 0 0 0-7.07-7.07L11.5 4.5"/>'
        '<path d="M14 11a5 5 0 0 0-7.07 0L4.1 13.83a5 5 0 0 0 7.07 7.07L12.5 19.5"/>'
        "</svg>"
    ),
    "check_circle": (
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>'
        "</svg>"
    ),
    "warning": (
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
        '<line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>'
        "</svg>"
    ),
}


def icon_html(name: str, color: str = "currentColor") -> str:
    """Return one of ``ICONS`` wrapped so it inherits ``color``, ready to drop into an f-string."""
    return f'<span style="display:inline-flex;color:{color};">{ICONS[name]}</span>'


def apply_theme() -> None:
    """
    Inject shared CSS for a consistent, modern look. Call once near the top
    of every page (after ``st.set_page_config``). Purely additive — every
    rule targets Streamlit's own stable component hooks (``data-testid``
    attributes, or documented classes like ``.stButton``), so a rule that no
    longer matches in some future Streamlit version just does nothing rather
    than breaking anything.
    """
    st.markdown(
        """
        <style>
        /* ── Typography ──────────────────────────────────────────────────── */
        h1, h2, h3 { letter-spacing: -0.01em; }
        [data-testid="stCaptionContainer"] { opacity: 0.75; }

        /* ── Buttons: rounded, subtle lift on hover ──────────────────────── */
        .stButton > button {
            border-radius: 8px;
            transition: transform 0.06s ease, box-shadow 0.15s ease;
            font-weight: 600;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 10px rgba(0,0,0,0.12);
        }
        .stButton > button[kind="primary"] {
            background: #522D80;
            border-color: #522D80;
        }
        .stButton > button[kind="primary"]:hover {
            background: #603699;
            border-color: #603699;
        }

        /* ── Metrics: bigger value, calmer label ─────────────────────────── */
        [data-testid="stMetricValue"] {
            font-size: 1.6rem !important;
            font-weight: 700;
        }
        [data-testid="stMetricLabel"] {
            opacity: 0.7;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }

        /* ── Expanders: card-like, rounded ────────────────────────────────── */
        [data-testid="stExpander"] {
            border-radius: 10px;
            border: 1px solid rgba(127,127,127,0.2);
        }

        /* ── Bordered containers (st.container(border=True)): soft shadow ─── */
        [data-testid="stVerticalBlockBorderWrapper"] > div {
            border-radius: 12px;
        }

        /* ── Sidebar: tighter, calmer nav ─────────────────────────────────── */
        [data-testid="stSidebarNav"] { padding-top: 0.5rem; }
        .sidebar-brand {
            font-size: 1.3rem;
            font-weight: 700;
            color: #522D80;
            letter-spacing: -0.02em;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }
        .sidebar-sub {
            font-size: 0.75rem;
            opacity: 0.65;
            margin-bottom: 0.75rem;
        }

        /* ── Risk colour bands (shared text token) ───────────────────────── */
        .risk-low    { color: #2ecc71; font-weight: 700; }
        .risk-medium { color: #f39c12; font-weight: 700; }
        .risk-high   { color: #e74c3c; font-weight: 700; }
        </style>
        """,
        unsafe_allow_html=True,
    )
