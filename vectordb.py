from flask import Flask, request, jsonify
import chromadb
import pickle
from typing import List, Dict
import uuid
import os
from chromadb.config import Settings
from chromadb.errors import InvalidCollectionException
from functools import lru_cache

app = Flask(__name__)

# Configuration
PERSIST_DIR = os.getenv('PERSIST_DIR', 'chroma_db')
PICKLE_FILE = os.getenv('PICKLE_FILE', 'processed_videos.pickle')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'video_transcripts')

def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

@lru_cache(maxsize=1)
def get_persistent_client():
    """Initialize ChromaDB client with persistence."""
    return chromadb.PersistentClient(path=PERSIST_DIR)

def load_processed_videos(pickle_file: str) -> List[Dict]:
    """Load processed video chunks from pickle file."""
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)

def create_or_load_collection():
    """Create a new collection or load existing one with data verification."""
    chroma_client = get_persistent_client()
    
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        return collection
    except InvalidCollectionException:
        # Load chunks from pickle file
        chunks = load_processed_videos(PICKLE_FILE)
        
        # Create new collection
        collection = chroma_client.create_collection(name=COLLECTION_NAME)
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            documents.append(chunk['text'])
            
            metadata = {
                'video_title': chunk['video_title'],
                'video_url': chunk['video_url'],
                'video_id': chunk['video_id'],
                'start_time': chunk['start_time'],
                'end_time': chunk['end_time'],
                'timestamp': f"{format_timestamp(chunk['start_time'])} - {format_timestamp(chunk['end_time'])}"
            }
            
            metadatas.append(metadata)
            ids.append(chunk_id)
        
        # Add chunks to collection in batches
        batch_size = 500
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            collection.add(
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
        
        return collection

def query_collection(query_text: str, n_results: int = 3):
    """Query the collection and return formatted results."""
    collection = create_or_load_collection()
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    
    formatted_results = []
    for idx, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        formatted_results.append({
            'text': doc,
            'video_title': metadata['video_title'],
            'video_url': metadata['video_url'],
            'timestamp': metadata['timestamp'],
            'relevance_rank': idx + 1
        })
    
    return formatted_results

@app.route('/api/query', methods=['POST'])
def create_query():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    data = request.get_json()
    query_text = data.get('query')
    n_results = data.get('n_results', 3)
    
    if not query_text:
        return jsonify({"error": "Query text is required"}), 400
    
    try:
        results = query_collection(query_text, n_results)
        return jsonify({
            "status": "success",
            "results": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    # Ensure persist directory exists
    os.makedirs(PERSIST_DIR, exist_ok=True)
    
    # Initialize collection on startup
    create_or_load_collection()
    
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)