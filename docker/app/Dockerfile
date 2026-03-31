# ═══════════════════════════════════════════════════════════════════════════════
# RAGScope — Application Dockerfile
# Multi-stage build using uv for fast, reproducible dependency installation.
#
# Stages:
#   builder  — installs Python deps via uv into an isolated venv
#   app      — lean runtime image; copies only the venv and source
#
# Build:  docker build -f docker/app/Dockerfile -t ragscope:latest .
# ═══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
# Use the official uv image so we get a pinned, verified uv binary without
# needing to curl anything at build time.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# uv environment flags:
#   UV_COMPILE_BYTECODE  — pre-compiles .pyc files so the runtime container
#                          starts faster (compilation is free at build time).
#   UV_LINK_MODE=copy    — copies files instead of hardlinking; required when
#                          the venv and the uv cache are on different filesystems
#                          (which they are in Docker).
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies first — this layer is cached unless pyproject.toml or
# uv.lock changes, making rebuilds fast during active development.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
      --frozen \
      --no-install-project \
      --no-dev

# Copy the full source and install the project itself (editable = false).
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
      --frozen \
      --no-dev


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS app

# Non-root user for security — never run application code as root.
RUN groupadd --gid 1001 ragscope \
 && useradd  --uid 1001 --gid ragscope --shell /bin/bash --create-home ragscope

WORKDIR /app

# Copy the fully-populated venv from the builder; no uv or pip needed at runtime.
COPY --from=builder --chown=ragscope:ragscope /app/.venv /app/.venv
COPY --from=builder --chown=ragscope:ragscope /app       /app

# Prepend the venv bin directory so every `python` / `streamlit` invocation
# uses the venv without needing explicit activation.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=0

# Streamlit configuration — disable the welcome screen and browser auto-open,
# and bind to all interfaces so Docker can forward the port.
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_RUN_ON_SAVE=false \
    STREAMLIT_THEME_BASE=dark

EXPOSE 8501

# Health check — Streamlit exposes a /_stcore/health endpoint.
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" \
        || exit 1

USER ragscope

COPY --chown=ragscope:ragscope docker/scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]