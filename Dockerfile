# =============================================================
# Stochastic Volatility MLOps - Dockerfile
# Multi-stage build: slim production image with FastAPI service
# Author: Fidel Mehra
# =============================================================

# ---- Stage 1: Build dependencies ----
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---- Stage 2: Production image ----
FROM python:3.11-slim AS production

LABEL maintainer="Fidel Mehra"
LABEL description="Stochastic Volatility Forecasting & Risk Regime Detection API"
LABEL version="1.0.0"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy project source code
COPY config/ ./config/
COPY src/ ./src/
COPY app/ ./app/

# Create required directories
RUN mkdir -p data/raw data/processed models/registry

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')"

# Run the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
