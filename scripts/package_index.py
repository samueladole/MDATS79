"""
Package the local index as a release archive for scripts/fetch_index.py.

Maintainer tool. Bundles data/chroma_store, the BM25 corpus, the ingestion
manifest and the knowledge graph into one .tar.gz, then prints its SHA-256 to
paste into INDEX_SHA256 in scripts/fetch_index.py. Upload the archive as the
`ragscope-index-v1.tar.gz` asset of the `index-v1` GitHub release.

    uv run python scripts/package_index.py [output.tar.gz]
"""

from __future__ import annotations

import gzip
import hashlib
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

ENTRIES = [
    "chroma_store",
    "bm25_corpus.jsonl",
    "ingestion_manifest.json",
    "graph/graph.json",
    "graph/node_to_chunks.json",
    "graph/manifest.json",
]


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "ragscope-index-v1.tar.gz"
    missing = [e for e in ENTRIES if not (DATA_DIR / e).exists()]
    if missing:
        print("Missing from data/:", ", ".join(missing))
        return 1

    def _clean(info: tarfile.TarInfo) -> tarfile.TarInfo:
        # Fixed owner and mode, so the archive does not carry local account names.
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        return info

    # No filename and mtime 0 in the gzip header, and tarfile adds directory entries in sorted
    # order, so the same data always produces the same archive and checksum.
    with out.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w") as tar:
        for entry in ENTRIES:
            tar.add(DATA_DIR / entry, arcname=entry, filter=_clean)

    h = hashlib.sha256()
    with out.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")
    print(f"INDEX_SHA256 = \"{h.hexdigest()}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
