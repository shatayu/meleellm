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

# Copy the application
COPY app.py .

# Copy the pre-built database
COPY chroma_db/ ./chroma_db/

# Expose port
EXPOSE 10000

# Use Gunicorn with log level set to info
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--log-level", "debug", "--access-logfile", "-", "--error-logfile", "-", "--chdir", "/app", "app:app"]