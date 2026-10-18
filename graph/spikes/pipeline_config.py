"""
The working docling-graph configuration, established by spike S1.

Each setting below is load-bearing; the comments record what goes wrong without
it, because several failure modes are silent.
"""
from pathlib import Path

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
            "max_output_tokens": 4096,
            "context_limit": 32000,
            # Without these the extraction is NOT reproducible: the same
            # passage yielded 4 nodes on one run and 2 on the next.
            "generation": {"temperature": 0.0, "seed": 42},
        },
        "parallel_workers": 1,
        "output_dir": str(output_dir) if output_dir else "out",
        "dump_to_disk": dump,
    }
