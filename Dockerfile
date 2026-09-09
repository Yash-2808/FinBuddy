# Multi-stage / Production Dockerfile for FinBuddy
FROM python:3.11-slim

# Set working directory
WORKDIR /app

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

# Expose ports: 8501 (Streamlit Frontend) and 8000 (FastAPI Backend)
EXPOSE 8501 8000

# Default entrypoint starts Streamlit dashboard
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
