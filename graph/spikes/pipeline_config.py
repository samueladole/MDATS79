"""
The working docling-graph configuration, established by spike S1.

Each setting below is load-bearing; the comments record what goes wrong without
it, because several failure modes are silent.

It also holds every path the graph scripts use. They all come from the project
settings, so the builder writes exactly where the retriever reads.
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.settings import settings  # noqa: E402


def _in_repo(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


GRAPH_PATH = _in_repo(settings.graph_path)                    # data/graph/graph.json
NODE_CHUNKS_PATH = _in_repo(settings.graph_node_chunks_path)  # data/graph/node_to_chunks.json
GRAPH_DIR = GRAPH_PATH.parent
MANIFEST_PATH = GRAPH_DIR / "manifest.json"
SAMPLE_DIR = GRAPH_DIR / "sample"                 # one .md file per sampled chunk
CORPUS_PATH = REPO_ROOT / "data" / "bm25_corpus.jsonl"

# docling-graph reads the Ollama address from the environment rather than from
# our settings, so pass the project's value through. A value already set in the
# shell still wins.
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
