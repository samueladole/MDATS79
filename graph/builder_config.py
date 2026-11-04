"""
The working docling-graph configuration, established by spike S1.

Each setting below is load-bearing; the comments record what goes wrong without
it, because several failure modes are silent.

It also holds every path the graph scripts use. They all come from the project
settings, so the builder writes exactly where the retriever reads.
"""
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.settings import settings  # noqa: E402


def _in_repo(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


# The BioASQ benchmark's graph. graph/build_graph.py --kb documents switches
# every path to the documents knowledge base (see pipeline/knowledge_base.py).
GRAPH_PATH = _in_repo(settings.graph_path)                    # data/graph/graph.json
NODE_CHUNKS_PATH = _in_repo(settings.graph_node_chunks_path)  # data/graph/node_to_chunks.json
GRAPH_DIR = GRAPH_PATH.parent
MANIFEST_PATH = GRAPH_DIR / "manifest.json"
EXTRACTIONS_DIR = GRAPH_DIR / "extractions"       # one saved result per extracted chunk
WORK_DIR = GRAPH_DIR / "work"                     # one temporary file per chunk in flight
CORPUS_PATH = REPO_ROOT / "data" / "bm25_corpus.jsonl"

#: Size of the corpus-principled sample behind the published graph.
SAMPLE_SIZE = 300


def select_sample(n: int = SAMPLE_SIZE, corpus_path: Path = CORPUS_PATH) -> list[tuple[str, str]]:
    """The n-chunk sample, as (chunk_id, text), read straight from the corpus.

    Sampling is corpus-principled: an even stride across the whole corpus,
    with no reference to the benchmark queries (a subset seeded from gold
    passages would give an oracle graph that flatters any later retrieval
    comparison). The stride oversamples by 1.6x so that, after keeping only
    chunks of 200-2,000 characters, exactly n remain.

    This reproduces the 300 chunks of the published graph exactly. Results are
    ordered by file-safe chunk id, the order the builder has always used.
    """
    with corpus_path.open() as fh:
        total = sum(1 for _ in fh)
    stride = max(1, total // int(n * 1.6))
    picked: list[tuple[str, str]] = []
    with corpus_path.open() as fh:
        for i, line in enumerate(fh):
            if i % stride == 0:
                record = json.loads(line)
                if 200 < len(record["text"]) < 2000:
                    picked.append((record["id"], record["text"]))
            if len(picked) >= n:
                break
    return sorted(picked[:n], key=lambda p: p[0].replace("::", "__"))

# The Ollama address: the project's setting, unless the shell sets
# OLLAMA_BASE_URL. It is passed to docling-graph explicitly (llm_overrides
# below), because docling-graph 1.9.1 ignores the environment variable: its
# built-in default, http://localhost:11434, takes precedence over it.
os.environ.setdefault("OLLAMA_BASE_URL", settings.ollama_base_url)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

EXTRACTOR_MODEL = "llama3.2:latest"


def build_config(source_file: Path, *, model: str = EXTRACTOR_MODEL,
                 output_dir: Path | None = None, dump: bool = False) -> dict:
    """Config for extracting ONE chunk file into a graph.

    source_file must be a single file, never a directory: given a directory,
    docling-graph converts the directory PATH STRING into the document and
    extracts entities from it, reporting success with hallucinated nodes.
    """
    return {
        "source": str(source_file),
        "template": "biomed_template.BiomedicalAbstract",
        "backend": "llm",
        "inference": "local",
        "models": {
            "llm": {"local": {"model": model, "provider": "ollama"},
                    "remote": {"model": model, "provider": "ollama"}},
            "vlm": {"local": {"model": "numind/NuExtract-2.0-8B", "provider": "docling"}},
        },
        # one-to-one is page-based; a markdown DoclingDocument has ZERO pages,
        # so it completes with "0/0 pages" and no models. many-to-one works
        # over chunks instead.
        "processing_mode": "many-to-one",
        "extraction_contract": "direct",
        # Must be True for the chunk path to be used at all. Safe because one
        # input file is already one of our chunks, so at 4096 max tokens the
        # split is a no-op and the 1:1 chunk mapping holds.
        "use_chunking": True,
        "chunk_max_tokens": 4096,
        "llm_input_format": "markdown",
        "provenance": "detailed",
        "llm_overrides": {
            "connection": {"base_url": os.environ["OLLAMA_BASE_URL"]},
            # Default max_output_tokens is 131072, which exceeds the model's
            # context window and aborts before any LLM call is made.
            "max_output_tokens": 8192,
            "context_limit": 32000,
            # Without these the extraction is NOT reproducible: the same
            # passage yielded 4 nodes on one run and 2 on the next.
            "generation": {"temperature": 0.0, "seed": 42},
        },
        "parallel_workers": 1,
        # Only written to when dump=True.
        "output_dir": str(output_dir or GRAPH_DIR / "docling_dump"),
        "dump_to_disk": dump,
    }
