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

# Copy dependency list first to leverage caching layers
COPY requirements.txt .

# Install Python packages safely
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project items into the container directory
COPY . .

# Create directory skeletons for operational outputs
RUN mkdir -p logs outputs

# Expose internal API application port
EXPOSE 8000

# Launch production server using Uvicorn ASGI pointed directly to your main.py file
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
