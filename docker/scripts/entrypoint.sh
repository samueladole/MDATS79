#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# RAGScope — Container Entrypoint
#
# Responsibilities:
#   1. Wait for Ollama to be reachable and have at least one model loaded.
#   2. Wait for ChromaDB to be reachable.
#   3. Start the Streamlit dashboard (or any command passed as arguments).
#
# Usage (automatic via Docker ENTRYPOINT):
#   /entrypoint.sh                          → starts Streamlit dashboard
#   /entrypoint.sh uv run python foo.py     → runs an arbitrary command instead
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[entrypoint]${NC} $*"; }
warn()  { echo -e "${YELLOW}[entrypoint]${NC} $*"; }
error() { echo -e "${RED}[entrypoint]${NC} $*" >&2; }

# ── Configuration (override via environment) ──────────────────────────────────
OLLAMA_HOST="${OLLAMA_BASE_URL:-http://ollama:11434}"
CHROMA_HOST="${CHROMA_HOST:-chromadb}"
CHROMA_PORT="${CHROMA_PORT:-8000}"
MAX_WAIT="${MAX_WAIT_SECONDS:-120}"   # maximum seconds to wait per service

# ── Generic wait function ─────────────────────────────────────────────────────
wait_for() {
    local service="$1"
    local url="$2"
    local elapsed=0
    local interval=3

    info "Waiting for ${service} at ${url} …"
    until curl --silent --fail --max-time 3 "${url}" > /dev/null 2>&1; do
        if (( elapsed >= MAX_WAIT )); then
            error "Timed out after ${MAX_WAIT}s waiting for ${service}."
            exit 1
        fi
        warn "  ${service} not ready — retrying in ${interval}s (${elapsed}s elapsed)"
        sleep "${interval}"
        (( elapsed += interval ))
    done
    info "${service} is ready. ✓"
}

# ── Wait for Ollama ───────────────────────────────────────────────────────────
wait_for "Ollama" "${OLLAMA_HOST}/api/tags"

# Additionally confirm that at least one model is available.
# The ollama-init service pulls models, but may still be running on first boot.
info "Checking Ollama has models available …"
elapsed=0
until [[ $(curl --silent "${OLLAMA_HOST}/api/tags" | python3 -c \
    "import sys,json; d=json.load(sys.stdin); print(len(d.get('models',[])))" \
    2>/dev/null) -gt 0 ]]; do
    if (( elapsed >= MAX_WAIT )); then
        error "Timed out waiting for Ollama models. Is ollama-init running?"
        exit 1
    fi
    warn "  No models found yet — waiting for ollama-init to finish pulling (${elapsed}s elapsed)"
    sleep 5
    (( elapsed += 5 ))
done
info "Ollama models confirmed. ✓"

# ── Wait for ChromaDB ─────────────────────────────────────────────────────────
wait_for "ChromaDB" "http://${CHROMA_HOST}:${CHROMA_PORT}/api/v1/heartbeat"

# ── Start the application ─────────────────────────────────────────────────────
if [[ $# -gt 0 ]]; then
    # If arguments were passed (e.g. from `docker compose exec`), run them directly.
    info "Running command: $*"
    exec "$@"
else
    # Default: launch the Streamlit dashboard.
    info "Starting RAGScope dashboard on :${STREAMLIT_SERVER_PORT:-8501} …"
    exec streamlit run dashboard/app.py \
        --server.address="${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}" \
        --server.port="${STREAMLIT_SERVER_PORT:-8501}" \
        --server.headless=true
fi