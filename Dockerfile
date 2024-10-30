FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app.py and pickle file explicitly
COPY app.py .
COPY *.pickle .

# Create directory for ChromaDB persistence
RUN mkdir -p chroma_db

# Explicitly expose port
EXPOSE 10000

# Debug: List files
RUN ls -la

# Use module syntax for Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--chdir", "/app", "app:app"]