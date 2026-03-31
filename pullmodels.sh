#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# RAGScope — Ollama Model Puller
#
# Run once by the `ollama-init` Compose service on first boot.
# Pulls all models required for the research experiment into the shared
# /models volume so both `ollama` and subsequent `ollama-init` restarts
# find them pre-cached.
#
# Models are skipped if already present (Ollama is idempotent on re-pull,
# but we do an explicit check to avoid unnecessary API traffic).
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[pull_models]${NC} $*"; }
warn()  { echo -e "${YELLOW}[pull_models]${NC} $*"; }
error() { echo -e "${RED}[pull_models]${NC} $*" >&2; }

OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"

# ── Models required by the research experiment ────────────────────────────────
# Format: "name:tag"   — tag defaults to 'latest' if omitted.
MODELS=(
    "llama3:8b"     # Primary LLM — Condition A & B in the 2×2 factorial design
    "mistral:7b"    # Secondary LLM — Condition C & D
)

# ── Wait for Ollama daemon ────────────────────────────────────────────────────
info "Waiting for Ollama daemon at ${OLLAMA_HOST} …"
max_wait=60; elapsed=0
until curl --silent --fail --max-time 3 "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; do
    if (( elapsed >= max_wait )); then
        error "Ollama daemon not reachable after ${max_wait}s. Aborting."
        exit 1
    fi
    sleep 2; (( elapsed += 2 ))
done
info "Ollama daemon is up. ✓"

# ── Pull each model ───────────────────────────────────────────────────────────
for model in "${MODELS[@]}"; do
    # Check whether the model already exists in the local registry.
    if curl --silent "${OLLAMA_HOST}/api/tags" \
        | python3 -c "import sys,json; models=[m['name'] for m in json.load(sys.stdin).get('models',[])]; exit(0 if '${model}' in models else 1)" \
        2>/dev/null; then
        info "Model '${model}' already present — skipping pull."
    else
        info "Pulling model '${model}' …"
        # `ollama pull` streams progress; pipe through cat to preserve output.
        if ollama pull "${model}" \
            --ollama-host "${OLLAMA_HOST}" 2>&1 | cat; then
            info "Model '${model}' pulled successfully. ✓"
        else
            error "Failed to pull model '${model}'. Check disk space and network."
            exit 1
        fi
    fi
done

info "All models ready. ollama-init exiting cleanly."