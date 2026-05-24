# Base image: Official lightweight Python 3.11
FROM python:3.11-slim

# Set system optimizations
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory inside container
WORKDIR /app

# Install basic system health check dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list first to leverage caching
COPY requirements.txt .

# Install Python packages safely
RUN pip install --no-cache-dir -r requirements.txt

# Copy application layers, configurations, and trained models
COPY ./src ./src
COPY ./models ./models
COPY ./config.yaml .

# Create directory skeletons for operational outputs
RUN mkdir -p logs outputs

# Expose internal API application port
EXPOSE 8000

# Health check to monitor container operational state
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
  CMD curl -f http://localhost:8000/health || exit 1

# Launch production server using Uvicorn ASGI
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
