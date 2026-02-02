FROM ghcr.io/astral-sh/uv:latest AS uv_bin

# --- Stage 1: Build stage ---
FROM python:3.11-slim AS builder

COPY --from=uv_bin /uv /uvx /bin/

# Install git for cloning private repositories
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set uv configuration
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

# Configure git to use token for private repo access
ARG GITHUB_TOKEN
RUN if [ -n "$GITHUB_TOKEN" ]; then \
    cd /tmp && \
    echo "https://${GITHUB_TOKEN}@github.com" > /root/.git-credentials && \
    GIT_CONFIG_GLOBAL=/root/.gitconfig git config --global credential.helper store && \
    GIT_CONFIG_GLOBAL=/root/.gitconfig git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"; \
    fi

# Install dependencies first for caching
# We only need these files to sync the environment
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# --- Stage 2: Final stage ---
FROM python:3.11-slim

# Copy uv from the uv_bin stage
COPY --from=uv_bin /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Configure git to use token for private repo access in runtime
ARG GITHUB_TOKEN
RUN if [ -n "$GITHUB_TOKEN" ]; then \
    cd /tmp && \
    echo "https://${GITHUB_TOKEN}@github.com" > /root/.git-credentials && \
    GIT_CONFIG_GLOBAL=/root/.gitconfig git config --global credential.helper store && \
    GIT_CONFIG_GLOBAL=/root/.gitconfig git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"; \
    fi

# Copy the environment uv created (.venv is the default)
COPY --from=builder /app/.venv /app/.venv

# Add the venv to the path
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY . .

EXPOSE 7000

CMD ["uv", "run", "--upgrade", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7000"]