"""
RAGScope — Text Cleaner
========================
Normalises raw text from the BioASQ benchmark corpus before chunking
and embedding. Handles HTML artefacts, unicode noise, and whitespace
inconsistencies common in PubMed abstract text.
"""

from __future__ import annotations

import re
import unicodedata

# ── Public API ────────────────────────────────────────────────────────────────


def clean(text: str) -> str:
    """
    Apply the full cleaning pipeline to a raw text string.

    Steps applied in order:
        1. Unicode normalisation (NFC)
        2. HTML tag stripping
        3. Control character removal
        4. Whitespace normalisation
        5. Repeated punctuation collapsing
        6. Leading/trailing strip

    Parameters
    ----------
    text : str
        Raw input text from any corpus source.

    Returns
    -------
    str
        Cleaned text ready for chunking.
    """
    text = _normalise_unicode(text)
    text = _strip_html(text)
    text = _remove_control_chars(text)
    text = _normalise_whitespace(text)
    text = _collapse_repeated_punctuation(text)
    return text.strip()


def is_meaningful(text: str, min_chars: int = 50) -> bool:
    """
    Return True if the text is likely to be a useful passage for retrieval.

    Rejects:
        - Passages shorter than ``min_chars`` characters after cleaning
        - Passages that are mostly digits or punctuation (navigation artefacts)
        - Empty strings

    Parameters
    ----------
    text : str
        Cleaned text to evaluate.
    min_chars : int
        Minimum character length threshold. Default 50.
    """
    if not text or len(text) < min_chars:
        return False

    # Reject if >60 % of characters are non-alphabetic (e.g. tables, code blocks)
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count / len(text) < 0.40:
        return False

    return True


# ── Private helpers ───────────────────────────────────────────────────────────


def _normalise_unicode(text: str) -> str:
    """NFC normalisation ensures consistent representation of accented chars."""
    return unicodedata.normalize("NFC", text)


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode common HTML entities."""
    # Strip tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common entities
    entities = {
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&#39;": "'",
        "&nbsp;": " ",
        "&apos;": "'",
    }
    for entity, replacement in entities.items():
        text = text.replace(entity, replacement)
    return text


def _remove_control_chars(text: str) -> str:
    """Remove ASCII control characters (0x00–0x1F) except newline and tab."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


def _normalise_whitespace(text: str) -> str:
    """
    Collapse multiple whitespace characters into a single space,
    preserving paragraph boundaries (double newline → single newline).
    """
    # Collapse multiple blank lines to a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Replace tabs and carriage returns with a space
    text = re.sub(r"[\t\r]", " ", text)
    # Collapse multiple spaces into one
    text = re.sub(r" {2,}", " ", text)
    return text


def _collapse_repeated_punctuation(text: str) -> str:
    """Collapse sequences like '...' or '---' that add no semantic content."""
    text = re.sub(r"\.{3,}", "...", text)
    text = re.sub(r"-{3,}", "—", text)
    text = re.sub(r"={3,}", "", text)
    return text
