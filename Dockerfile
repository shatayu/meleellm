FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create directory for ChromaDB persistence
RUN mkdir -p chroma_db

CMD gunicorn --bind 0.0.0.0:$PORT app:app