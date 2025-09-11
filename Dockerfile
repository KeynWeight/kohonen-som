# Multi-stage Dockerfile for Kohonen SOM
# Supports both CLI and API deployments

# Base stage with common dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --shell /bin/bash --create-home app

# Set working directory
WORKDIR /app

# Install UV for fast dependency management
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy source code
COPY som/ ./som/
COPY cli.py api.py examples.py ./

# Change ownership to app user
RUN chown -R app:app /app

# Switch to app user
USER app

# Development stage
FROM base as development

# Install development dependencies
USER root
RUN uv sync --frozen
USER app

# API stage
FROM base as api

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

# CLI stage
FROM base as cli

# Set entrypoint for CLI usage
ENTRYPOINT ["python", "cli.py"]
CMD ["--help"]

# Production API stage with additional optimizations
FROM api as production

USER root

# Install production optimizations
RUN pip install gunicorn

USER app

# Use Gunicorn for production
CMD ["gunicorn", "api:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "120"]