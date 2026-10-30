"""
Download the prebuilt RAGScope index, so you can skip ingestion.

Building the index from source embeds 30,850 BioASQ passages, which takes
20-40 minutes on a CPU. This script instead downloads the index the published
results were produced with, checks its SHA-256, and unpacks it into data/:

    data/chroma_store/          vector index (open with CHROMA_MODE=embedded)
    data/bm25_corpus.jsonl      corpus for BM25 / hybrid retrieval
    data/ingestion_manifest.json
    data/graph/                 knowledge graph for graph retrieval

Usage
-----
    uv run python scripts/fetch_index.py            # download and unpack
    uv run python scripts/fetch_index.py --force    # replace an existing index

To rebuild from source instead: uv run python pipeline/ingestion.py
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

#: Release asset and its checksum. Written by scripts/package_index.py.
INDEX_URL = (
    "https://github.com/samueladole/MDATS79/releases/download/index-v1/"
    "ragscope-index-v1.tar.gz"
)
INDEX_SHA256 = "2aeba9ad8ddaa8ce0f404726db56e2d216fb020584f44124ba6a0ee9ddf10b1f"

#: Top-level entries the archive may contain. Anything else is refused.
EXPECTED = {"chroma_store", "bm25_corpus.jsonl", "ingestion_manifest.json", "graph"}


def _download(url: str, dest: Path) -> None:
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as resp, dest.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--force", action="store_true", help="replace an existing index")
    parser.add_argument("--archive", type=Path, help="use a local archive instead of downloading")
    args = parser.parse_args()

    existing = [DATA_DIR / name for name in EXPECTED if (DATA_DIR / name).exists()]
    if existing and not args.force:
        print("An index already exists:", *(f"  {p}" for p in existing), sep="\n")
        print("Re-run with --force to replace it.")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        archive = args.archive or Path(tmp) / "index.tar.gz"
        if not args.archive:
            _download(INDEX_URL, archive)

        print("Checking SHA-256 …")
        digest = _sha256(archive)
        if digest != INDEX_SHA256:
            print(f"Checksum mismatch: expected {INDEX_SHA256}, got {digest}.")
            return 1

        with tarfile.open(archive, "r:gz") as tar:
            tops = {m.name.split("/", 1)[0] for m in tar.getmembers()}
            if not tops <= EXPECTED:
                print(f"Unexpected entries in archive: {sorted(tops - EXPECTED)}")
                return 1
            for name in tops:
                target = DATA_DIR / name
                if target.is_dir():
                    shutil.rmtree(target)
                elif target.exists():
                    target.unlink()
            # filter="data" refuses absolute paths, links out of the tree and
            # device files, so a tampered archive cannot write outside data/.
            tar.extractall(DATA_DIR, filter="data")

    print(f"Index unpacked into {DATA_DIR}.")
    print("Set CHROMA_MODE=embedded in .env (the default in .env.example) to use it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
