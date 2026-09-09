# Multi-stage / Production Dockerfile for FinBuddy
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app:$PYTHONPATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose ports
EXPOSE 8501 8000 10000

# Default entrypoint starts Streamlit dashboard and binds to Render's dynamic $PORT or 8501
CMD sh -c "streamlit run frontend/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"
