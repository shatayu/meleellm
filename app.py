from flask import Flask, request, jsonify
import chromadb
import pickle
from typing import List, Dict
import uuid
import os
from chromadb.config import Settings
from chromadb.errors import InvalidCollectionException
from functools import lru_cache
import traceback

app = Flask(__name__)

# Configuration
PERSIST_DIR = os.getenv('PERSIST_DIR', 'chroma_db')
PICKLE_FILE = os.getenv('PICKLE_FILE', 'processed_videos.pkl')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'video_transcripts')

print(f"Starting application with:")
print(f"PERSIST_DIR: {PERSIST_DIR}")
print(f"PICKLE_FILE: {PICKLE_FILE}")
print(f"COLLECTION_NAME: {COLLECTION_NAME}")

def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

@lru_cache(maxsize=1)
def get_persistent_client():
    """Initialize ChromaDB client with persistence."""
    print("\n=== Getting ChromaDB Client ===")
    print(f"Creating ChromaDB client with persist_dir: {PERSIST_DIR}")
    print(f"Persist directory exists: {os.path.exists(PERSIST_DIR)}")
    if os.path.exists(PERSIST_DIR):
        print(f"Persist directory contents: {os.listdir(PERSIST_DIR)}")
    try:
        client = chromadb.PersistentClient(path=PERSIST_DIR)
        print("Successfully created ChromaDB client")
        return client
    except Exception as e:
        print(f"ERROR creating ChromaDB client: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        raise

def load_processed_videos(pickle_file: str) -> List[Dict]:
    """Load processed video chunks from pickle file."""
    print("\n=== Loading Pickle File ===")
    print(f"Attempting to load pickle file: {pickle_file}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Current directory contents: {os.listdir()}")
    print(f"File exists: {os.path.exists(pickle_file)}")
    if os.path.exists(pickle_file):
        print(f"File size: {os.path.getsize(pickle_file)} bytes")
    
    try:
        with open(pickle_file, 'rb') as f:
            print("Successfully opened pickle file")
            chunks = pickle.load(f)
            print(f"Successfully loaded {len(chunks)} chunks")
            print(f"First chunk keys: {list(chunks[0].keys()) if chunks else 'No chunks'}")
            return chunks
    except FileNotFoundError:
        print(f"ERROR: Could not find file {pickle_file}")
        raise
    except Exception as e:
        print(f"ERROR loading pickle file: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        raise

def create_or_load_collection():
    """Create a new collection or load existing one with data verification."""
    print("\n=== Creating/Loading Collection ===")
    print("Starting create_or_load_collection")
    
    try:
        print("Getting ChromaDB client...")
        chroma_client = get_persistent_client()
        print("Successfully got ChromaDB client")
        
        try:
            print(f"Attempting to get existing collection: {COLLECTION_NAME}")
            collection = chroma_client.get_collection(name=COLLECTION_NAME)
            doc_count = collection.count()
            print(f"Found existing collection with {doc_count} documents")
            return collection
        except (InvalidCollectionException, ValueError) as e:
            print(f"Collection doesn't exist, got error: {str(e)}")
            print("Will attempt to create new collection...")
            
            try:
                # Load chunks from pickle file
                print("Loading chunks from pickle file...")
                chunks = load_processed_videos(PICKLE_FILE)
                print(f"Successfully loaded {len(chunks)} chunks")
                
                # Create new collection
                print("Creating new collection...")
                collection = chroma_client.create_collection(name=COLLECTION_NAME)
                print("Successfully created new empty collection")
                
                # Prepare data for ChromaDB
                print("Preparing documents for insertion...")
                documents = []
                metadatas = []
                ids = []
                
                for i, chunk in enumerate(chunks):
                    if i == 0:
                        print(f"Processing first chunk: {chunk.keys()}")
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
                    print(f"Adding batch {i} to {end_idx}...")
                    collection.add(
                        documents=documents[i:end_idx],
                        metadatas=metadatas[i:end_idx],
                        ids=ids[i:end_idx]
                    )
                    print(f"Successfully added batch {i} to {end_idx}")
                
                print("Successfully created and populated collection")
                return collection
                
            except Exception as inner_e:
                print(f"ERROR while creating new collection: {str(inner_e)}")
                print(f"Full traceback: {traceback.format_exc()}")
                raise
            
    except Exception as e:
        print(f"ERROR in create_or_load_collection: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        raise

def query_collection(query_text: str, n_results: int = 3):
    """Query the collection and return formatted results."""
    print("\n=== Querying Collection ===")
    print(f"Query text: {query_text}")
    print(f"Number of results requested: {n_results}")
    
    collection = create_or_load_collection()
    print(f"Got collection, count: {collection.count()}")
    
    print("Executing query...")
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    print("Raw query results:", results)  # Add this to see full query response
    print(f"Got {len(results['documents'][0]) if results['documents'] else 0} results")
    
    if not results['documents'][0]:
        print("No documents found in query results")
        return []
        
    formatted_results = []
    for idx, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        result = {
            'text': doc,
            'video_title': metadata['video_title'],
            'video_url': metadata['video_url'],
            'timestamp': metadata['timestamp'],
            'relevance_rank': idx + 1
        }
        print(f"Formatted result {idx + 1}:", result)
        formatted_results.append(result)
    
    print("Successfully formatted results")
    return formatted_results

@app.route('/api/query')
def get_query():
    print("\n=== Query Endpoint Called ===")
    query_text = request.args.get('query')
    n_results = request.args.get('n_results', default=3, type=int)
    
    print(f"Received query: {query_text}")
    print(f"Requested results: {n_results}")
    
    if not query_text:
        print("Error: No query text provided")
        return jsonify({"error": "Query parameter is required"}), 400
    
    try:
        print("Calling query_collection...")
        results = query_collection(query_text, n_results)
        print(f"Query returned {len(results)} results")
        print("Full results:", results)
        return jsonify({
            "status": "success",
            "results": results
        })
    except Exception as e:
        print(f"Error in query endpoint: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/health')
def health_check():
    print("\n=== Health Check Called ===")
    try:
        print("Starting health check")
        collection = create_or_load_collection()
        count = collection.count()
        print(f"Health check successful. Collection size: {count}")
        return jsonify({
            "status": "healthy",
            "collection_size": count,
            "persist_dir": PERSIST_DIR,
            "pickle_file": PICKLE_FILE,
            "collection_name": COLLECTION_NAME
        }), 200
    except Exception as e:
        print("Health check failed!")
        print(f"Error details: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "persist_dir": PERSIST_DIR,
            "pickle_file": PICKLE_FILE,
            "collection_name": COLLECTION_NAME
        }), 500

if __name__ == '__main__':
    # Initial setup
    print("\n=== Application Starting ===")
    
    # Ensure persist directory exists
    os.makedirs(PERSIST_DIR, exist_ok=True)
    print(f"Ensured persist directory exists: {PERSIST_DIR}")
    print(f"Persist directory contents: {os.listdir(PERSIST_DIR)}")
    
    # Initialize collection on startup
    try:
        print("Initializing collection on startup...")
        collection = create_or_load_collection()
        print("Successfully initialized collection on startup")
    except Exception as e:
        print(f"Failed to initialize collection on startup: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
    
    port = int(os.getenv('PORT', 10000))
    print(f"\nStarting Flask application on port {port}")
    app.run(host='0.0.0.0', port=port)