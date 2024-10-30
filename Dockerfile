FROM python:3.10.15-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir numpy==1.24.3  # Explicitly install older numpy

COPY app.py .
COPY *.pickle .

# Create directory for ChromaDB persistence
RUN mkdir -p chroma_db

# Explicitly expose port
EXPOSE 10000

# Debug: List files and installed packages
RUN ls -la && pip freeze

CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--chdir", "/app", "app:app"]