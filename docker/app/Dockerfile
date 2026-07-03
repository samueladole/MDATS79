# ═══════════════════════════════════════════════════════════════════════════════
# RAGScope — Application Dockerfile
# Multi-stage build using uv for fast, reproducible dependency installation.
#
# Runtime runs as ROOT intentionally to avoid Docker volume permission issues
# with mounted telemetry / experiment directories.
# ═══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Install locked dependencies first (cache-friendly)
COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --all-extras --frozen --no-install-project --no-dev

# Copy source code
COPY . .

# Install project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Remove caches / pyc
RUN find /app/.venv -type d -name "__pycache__" -exec rm -rf {} + \
 && find /app/.venv -type f -name "*.pyc" -delete


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS app

# Install runtime tools
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        curl \
        bash \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy built app + virtualenv
COPY --from=builder /app /app

# Copy entrypoint
COPY docker/scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Ensure writable directories exist
RUN mkdir -p /app/telemetry/store \
 && mkdir -p /app/experiments/results \
 && chmod -R 777 /app/telemetry \
 && chmod -R 777 /app/experiments

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

# Intentionally run as root
ENTRYPOINT ["/entrypoint.sh"]