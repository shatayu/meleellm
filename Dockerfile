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
    pip install --no-cache-dir numpy==1.24.3

# Copy specifically the files we need
COPY app.py .
COPY processed_videos.pkl .

# Create directory for ChromaDB persistence
RUN mkdir -p chroma_db

# Debug: List files and installed packages
RUN ls -la && pip freeze

# Expose port
EXPOSE 10000

CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--chdir", "/app", "app:app"]