# The Sanctuary - Multi-stage Docker build
# Build: docker build -t sanctuary .
# Run:   docker run -it --rm sanctuary

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.11-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip wheel setuptools

WORKDIR /app
COPY pyproject.toml setup.py ./

# Install PyTorch CPU first (large download, cache separately)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install project dependencies
RUN pip install --no-cache-dir . || \
    pip install --no-cache-dir \
    numpy transformers sentence-transformers langchain chromadb accelerate \
    quart hypercorn httpx aiohttp "discord.py" python-dotenv pydantic

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.11-slim-bookworm AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg curl && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 sanctuary && \
    useradd --uid 1000 --gid sanctuary --shell /bin/bash --create-home sanctuary

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY --chown=sanctuary:sanctuary . .

RUN mkdir -p /app/data/memories /app/data/chroma /app/data/checkpoints /app/data/logs && \
    chown -R sanctuary:sanctuary /app/data

USER sanctuary

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/sanctuary:/app \
    SANCTUARY_BASE_DIR=/app/data \
    SANCTUARY_CHROMA_DIR=/app/data/chroma \
    SANCTUARY_LOG_DIR=/app/data/logs

EXPOSE 8000

# Health check - verify Python and mind module work
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import mind" || exit 1

CMD ["python", "-m", "sanctuary.run_cognitive_core"]
