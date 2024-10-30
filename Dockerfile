FROM python:3.10.15-slim

# Set Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1

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

# Use Gunicorn with log level set to info and access logging enabled
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--log-level", "debug", "--access-logfile", "-", "--error-logfile", "-", "--chdir", "/app", "app:app"]