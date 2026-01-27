FROM ghcr.io/astral-sh/uv:latest AS uv_bin

# --- Stage 1: Build stage ---
FROM python:3.11-slim AS builder

COPY --from=uv_bin /uv /uvx /bin/

# Set uv configuration
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies first for caching
# We only need these files to sync the environment
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev --extra cpu

# --- Stage 2: Final stage ---
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy the environment uv created (.venv is the default)
COPY --from=builder /app/.venv /app/.venv

# Add the venv to the path
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY . .

EXPOSE 7000

CMD ["fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "7000"]