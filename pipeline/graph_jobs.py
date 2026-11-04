"""
Graph builds in the background, for the dashboard.

The dashboard starts ``graph/build_graph.py --kb <name>`` as a separate
process, so a build of several minutes (your documents) or hours (BioASQ)
never blocks a page. The builder writes its process id to ``build.pid`` in
the knowledge base's graph folder and its output to ``build.log``. Progress is
read from the saved per-chunk extractions, so it is correct whether the build
was started here or from the command line.
"""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from pipeline.knowledge_base import PROJECT_ROOT, KnowledgeBase

BUILDER = PROJECT_ROOT / "graph" / "build_graph.py"

#: Builds started by this process, so finished ones are reaped and not
#: mistaken for running ones.
_PROCS: dict[int, subprocess.Popen] = {}


def docling_installed() -> bool:
    """Whether the optional docling-graph extra is installed here."""
    return importlib.util.find_spec("docling_graph") is not None


def _pid_file(kb: KnowledgeBase) -> Path:
    return kb.graph_dir / "build.pid"


def log_path(kb: KnowledgeBase) -> Path:
    return kb.graph_dir / "build.log"


def _alive(pid: int) -> bool:
    proc = _PROCS.get(pid)
    if proc is not None:
        return proc.poll() is None
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def running_pid(kb: KnowledgeBase) -> int | None:
    """PID of the build running for ``kb``'s graph, or None."""
    try:
        pid = int(_pid_file(kb).read_text().strip())
    except (OSError, ValueError):
        return None
    return pid if _alive(pid) else None


@dataclass
class BuildStatus:
    running: bool
    extracted: int  # chunks with a saved extraction (succeeded or failed)
    total: int      # chunks in the knowledge base's corpus
    log_tail: str


def build_status(kb: KnowledgeBase, tail_lines: int = 6) -> BuildStatus:
    ids: list[str] = []
    if kb.bm25_path.exists():
        with kb.bm25_path.open() as fh:
            ids = [json.loads(line)["id"] for line in fh if line.strip()]
    saved = kb.graph_dir / "extractions"
    extracted = sum(1 for cid in ids
                    if (saved / f"{cid.replace('::', '__')}.json").exists())
    log = log_path(kb)
    tail = ""
    if log.exists():
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-tail_lines:])
    return BuildStatus(running=running_pid(kb) is not None, extracted=extracted,
                       total=len(ids), log_tail=tail)


def start_build(kb: KnowledgeBase) -> str:
    """Start building ``kb``'s graph in the background. Returns a status message."""
    if not docling_installed():
        return ("The graph builder needs the optional docling-graph extra, which is not "
                "installed here. Install it with `uv sync --extra graph`.")
    if running_pid(kb):
        return "A graph build is already running for this knowledge base."
    if not kb.bm25_path.exists():
        return "Index the documents first: the graph is built from the indexed chunks."
    kb.graph_dir.mkdir(parents=True, exist_ok=True)
    with log_path(kb).open("w") as log:
        proc = subprocess.Popen(
            [sys.executable, str(BUILDER), "--kb", kb.name],
            cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
    _PROCS[proc.pid] = proc
    # Written here as well as by the builder, so a second click cannot start
    # another build before the first has loaded docling-graph.
    _pid_file(kb).write_text(str(proc.pid))
    return "started"


def stop_build(kb: KnowledgeBase) -> bool:
    """Stop a running build at once. Saved extractions are kept."""
    pid = running_pid(kb)
    if pid is None:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return False
    proc = _PROCS.pop(pid, None)
    if proc is not None:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    _pid_file(kb).unlink(missing_ok=True)
    return True
