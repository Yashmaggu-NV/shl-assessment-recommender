# SHL Assessment Recommender — Docker image
# Multi-stage build for a lean production image.

# ---------------------------------------------------------------------------
# Stage 1: Builder — install deps in a venv
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system deps needed for faiss-cpu and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: Runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY --chown=appuser:appuser . .

# Build FAISS index at image build time (baked in for fast cold starts)
# This requires sentence-transformers to download the model during build.
# Comment out if you prefer to build the index at container startup.
RUN python embeddings/build_index.py

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
