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
PICKLE_FILE = os.getenv('PICKLE_FILE', 'processed_videos.pkl')
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
    print(f"Creating ChromaDB client with persist_dir: {PERSIST_DIR}")
    return chromadb.PersistentClient(path=PERSIST_DIR)

def load_processed_videos(pickle_file: str) -> List[Dict]:
    """Load processed video chunks from pickle file."""
    print(f"Attempting to load pickle file: {pickle_file}")
    print(f"Current directory contents: {os.listdir()}")
    
    try:
        with open(pickle_file, 'rb') as f:
            chunks = pickle.load(f)
            print(f"Successfully loaded {len(chunks)} chunks")
            return chunks
    except FileNotFoundError:
        print(f"ERROR: Could not find file {pickle_file}")
        raise
    except Exception as e:
        print(f"ERROR loading pickle file: {str(e)}")
        raise

def create_or_load_collection():
    """Create a new collection or load existing one with data verification."""
    print("Starting create_or_load_collection")
    chroma_client = get_persistent_client()
    
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        print(f"Found existing collection with {collection.count()} documents")
        return collection
    except InvalidCollectionException:
        print("Collection doesn't exist, creating new one")
        try:
            # Load chunks from pickle file
            chunks = load_processed_videos(PICKLE_FILE)
            
            # Create new collection
            collection = chroma_client.create_collection(name=COLLECTION_NAME)
            print("Created new collection")
            
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
            
            print(f"Prepared {len(documents)} documents for insertion")
            
            # Add chunks to collection in batches
            batch_size = 500
            for i in range(0, len(documents), batch_size):
                end_idx = min(i + batch_size, len(documents))
                collection.add(
                    documents=documents[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    ids=ids[i:end_idx]
                )
                print(f"Added batch of chunks {i} to {end_idx}")
            
            print("Successfully created and populated collection")
            return collection
        except Exception as e:
            print(f"ERROR in create_or_load_collection: {str(e)}")
            raise

@app.route('/api/query')
def get_query():
    query_text = request.args.get('query')
    n_results = request.args.get('n_results', default=3, type=int)
    
    if not query_text:
        return jsonify({"error": "Query parameter is required"}), 400
    
    try:
        results = query_collection(query_text, n_results)
        return jsonify({
            "status": "success",
            "results": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health')
def health_check():
    try:
        print("Starting health check")
        collection = create_or_load_collection()
        count = collection.count()
        print(f"Health check successful. Collection size: {count}")
        return jsonify({
            "status": "healthy",
            "collection_size": count
        }), 200
    except Exception as e:
        print(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    # Ensure persist directory exists
    os.makedirs(PERSIST_DIR, exist_ok=True)
    print(f"Ensured persist directory exists: {PERSIST_DIR}")
    
    # Initialize collection on startup
    try:
        collection = create_or_load_collection()
        print(f"Initialized collection on startup")
    except Exception as e:
        print(f"Failed to initialize collection on startup: {str(e)}")
    
    port = int(os.getenv('PORT', 10000))
    app.run(host='0.0.0.0', port=port)