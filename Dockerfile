# =============================================================================
# HUMMINGBIRD-LEA - Dockerfile
# Powered by CoDre-X | B & D Servicing LLC
# =============================================================================
# Multi-stage build for optimized production image
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Build stage
# -----------------------------------------------------------------------------
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Production stage
# -----------------------------------------------------------------------------
FROM python:3.11-slim as production

# Labels
LABEL maintainer="B & D Servicing LLC"
LABEL description="Hummingbird-LEA - AI Assistant Platform"
LABEL version="1.0.0"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    APP_ENV=production \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # For EasyOCR
    libgl1 \
    libglib2.0-0 \
    # For health checks
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder
COPY --from=builder /usr/local /usr/local
ENV PATH=/root/.local/bin:$PATH

# Create non-root user for security
RUN groupadd -r hummingbird && useradd -r -g hummingbird hummingbird

# Create data directories
RUN mkdir -p /app/data/uploads \
             /app/data/knowledge \
             /app/data/memory \
             /app/data/templates \
             /app/data/logs \
    && chown -R hummingbird:hummingbird /app/data

# Copy application code
COPY --chown=hummingbird:hummingbird . .

# Switch to non-root user
USER hummingbird

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health/ || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "apps.hummingbird.main:app", "--host", "0.0.0.0", "--port", "8000"]
