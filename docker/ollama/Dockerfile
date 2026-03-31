# ═══════════════════════════════════════════════════════════════════════════════
# RAGScope — Ollama Service Dockerfile
#
# Extends the official Ollama image with:
#   - A non-root runtime user
#   - A persistent model cache directory with correct permissions
#   - A health-check endpoint probe
#
# Model pulling is intentionally NOT performed here — it is handled by the
# separate `ollama-init` one-shot service in docker-compose.yml. This keeps
# the image layer cache clean and the pull progress visible in Compose logs.
#
# GPU support: if a CUDA-capable GPU is present and the NVIDIA Container
# Toolkit is installed, set OLLAMA_ENABLE_CUDA=1 in your .env. The base
# ollama/ollama image ships both CPU and GPU inference; no separate image
# is required.
# ═══════════════════════════════════════════════════════════════════════════════

FROM ollama/ollama:latest

# Persist downloaded models to a named Docker volume so they survive container
# recreation. The OLLAMA_MODELS env var tells Ollama where to store them.
ENV OLLAMA_MODELS=/models \
    OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_KEEP_ALIVE=24h

RUN mkdir -p /models && chmod 777 /models

EXPOSE 11434

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
    CMD curl --silent --fail http://localhost:11434/api/tags || exit 1