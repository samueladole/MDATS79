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
# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Install deps first (cached layer)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --all-extras --frozen --no-install-project --no-dev

# Copy source and install project
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ── Strip runtime-irrelevant bloat from the venv ──────────────────────────────
RUN find /app/.venv -name "*.dist-info"   -type d -exec rm -rf {} + 2>/dev/null; \
    find /app/.venv -name "*.pyi"                  -delete          2>/dev/null; \
    find /app/.venv -name "tests"         -type d -path "*/site-packages/*" -exec rm -rf {} + 2>/dev/null; \
    find /app/.venv -name "test"          -type d -path "*/site-packages/*" -exec rm -rf {} + 2>/dev/null; \
    find /app/.venv -name "docs"          -type d -path "*/site-packages/*" -exec rm -rf {} + 2>/dev/null; \
    find /app/.venv -name "benchmarks"    -type d -path "*/site-packages/*" -exec rm -rf {} + 2>/dev/null; \
    find /app/.venv -name "examples"      -type d -path "*/site-packages/*" -exec rm -rf {} + 2>/dev/null; \
    find /app/.venv -name "*.h"           -delete 2>/dev/null; \
    true  # never fail the build


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS app

# curl is required by the HEALTHCHECK — install it cleanly
RUN groupadd --gid 1001 ragscope \
 && useradd --uid 1001 --gid ragscope --shell /bin/bash --create-home ragscope \
 && apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ✅ Single copy of /app (which already contains .venv) — no double-copy
COPY --from=builder --chown=ragscope:ragscope /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_RUN_ON_SAVE=false \
    STREAMLIT_THEME_BASE=dark

EXPOSE 8501

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

COPY --chown=ragscope:ragscope docker/scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER ragscope

ENTRYPOINT ["/entrypoint.sh"]