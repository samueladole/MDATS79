"""
Download a few open-access biomedical papers to try document ingestion with.

Fetches five CC BY articles on SGLT2 inhibitors from the PubMed Central
open-access dataset (AWS Open Data, no account needed) into
data/documents/examples/, checks each PDF against the MD5 that PMC publishes,
and writes ATTRIBUTION.md with the citation and licence of each.

    uv run python scripts/fetch_example_documents.py
    uv run python pipeline/ingestion.py --corpus documents

Nothing is committed to the repository: the PDFs stay on your machine.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "documents" / "examples"
BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com"

#: CC BY articles on SGLT2 inhibition, a topic the BioASQ benchmark also covers.
PMCIDS = ["PMC3563635", "PMC9849317", "PMC8187349", "PMC12457179", "PMC12209339"]


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    credits = []
    for pmcid in PMCIDS:
        meta = json.loads(_get(f"{BUCKET}/{pmcid}.1/{pmcid}.1.json"))
        if meta.get("license_code") != "CC BY" or meta.get("is_retracted"):
            print(f"Skipped {pmcid}: licence {meta.get('license_code')}, "
                  f"retracted={meta.get('is_retracted')}.")
            continue
        key, _, md5 = meta["pdf_url"].removeprefix("s3://pmc-oa-opendata/").partition("?md5=")
        target = OUT_DIR / f"{pmcid}.pdf"
        if not target.exists():
            data = _get(f"{BUCKET}/{key}")
            if md5 and hashlib.md5(data).hexdigest() != md5:  # noqa: S324 - PMC publishes MD5
                print(f"Skipped {pmcid}: checksum mismatch.")
                continue
            target.write_bytes(data)
        print(f"{pmcid}: {meta['title'][:80]}")
        credits.append(f"- **{pmcid}.pdf**: {meta['title']} {meta['citation']} "
                       f"Licence: {meta['license_code']}. https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/")

    (OUT_DIR / "ATTRIBUTION.md").write_text(
        "# Example documents\n\n"
        "Open-access articles from PubMed Central, redistributed under their "
        "Creative Commons Attribution (CC BY) licences.\n\n" + "\n".join(credits) + "\n"
    )
    print(f"\n{len(credits)} papers in {OUT_DIR}. Index them with:\n"
          "  uv run python pipeline/ingestion.py --corpus documents")
    return 0 if credits else 1


if __name__ == "__main__":
    sys.exit(main())
