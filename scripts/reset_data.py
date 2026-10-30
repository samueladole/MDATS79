"""
Clear your documents or the knowledge graph, so you can rebuild them.

    uv run python scripts/reset_data.py --documents          # documents index
    uv run python scripts/reset_data.py --graph              # knowledge graph
    uv run python scripts/reset_data.py --documents --graph  # both

Options
-------
    --documents             empty the "My documents" collection, its BM25 file and manifest
    --delete-document-files also delete the uploaded files in data/documents/
    --graph                 remove the graph files graph retrieval reads
                            (entities, relations, chunks covered)
    --delete-extractions    also delete the saved per-chunk extractions, so the
                            next build re-extracts everything (hours to days)
    --delete-sample         also delete the 300-chunk sample in data/graph/sample/
    --yes                   do not ask for confirmation

Nothing here touches the BioASQ benchmark index. To rebuild:

    documents: Knowledge Base page -> Upload documents -> Save and index,
               or uv run python pipeline/ingestion.py --corpus documents
    graph:     uv run python graph/build_graph.py           (300-chunk sample, ~1 h)
               uv run python graph/build_graph.py --all     (whole corpus, ~100 h)
               uv run python graph/build_graph.py --publish-only  (from saved extractions)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Clear your documents or the knowledge graph.")
    parser.add_argument("--documents", action="store_true")
    parser.add_argument("--delete-document-files", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--delete-extractions", action="store_true")
    parser.add_argument("--delete-sample", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    if not (args.documents or args.graph):
        parser.print_help()
        return 1

    plan = []
    if args.documents:
        plan.append("your documents' index (ChromaDB collection, BM25 file, manifest)"
                    + (" AND the uploaded files" if args.delete_document_files else ""))
    if args.graph:
        plan.append("the knowledge graph (graph, node-to-chunk index, manifest)"
                    + (" AND the saved extractions" if args.delete_extractions else "")
                    + (" AND the sample" if args.delete_sample else ""))
    print("This will delete:\n  - " + "\n  - ".join(plan))
    if not args.yes and input("Continue? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Cancelled.")
        return 1

    from loguru import logger

    logger.remove()
    from pipeline.maintenance import clear_documents, clear_graph

    reports = []
    if args.documents:
        reports.append(clear_documents(delete_files=args.delete_document_files))
    if args.graph:
        reports.append(clear_graph(delete_extractions=args.delete_extractions,
                                   delete_sample=args.delete_sample))
    for r in reports:
        for item in r.removed:
            print(f"removed  {item}")
        for item in r.kept:
            print(f"kept     {item}")
    print("Done. Restart the dashboard, or reload its page, to see the change.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
