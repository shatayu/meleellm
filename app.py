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
import anthropic
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Configuration
PERSIST_DIR = os.getenv('PERSIST_DIR', 'chroma_db')
PICKLE_FILE = os.getenv('PICKLE_FILE', 'processed_videos.pkl')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'video_transcripts')
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')


print(f"Starting application with:")
print(f"PERSIST_DIR: {PERSIST_DIR}")
print(f"PICKLE_FILE: {PICKLE_FILE}")
print(f"COLLECTION_NAME: {COLLECTION_NAME}")
print(f"CLAUDE API KEY: {CLAUDE_API_KEY}")

# Initialize Claude client
def process_with_claude(query: str, vector_results: List[Dict]) -> str:
    """Process vector search results through Claude."""
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    
    # Create context from vector results
    context = "\n\n".join([
        f"From video '{result['video_title']}' at {result['timestamp']}:\n{result['text']}"
        for result in vector_results
    ])
    
    prompt = f"""You are helping answer questions about Super Smash Bros Melee using video transcripts. 
The following are relevant excerpts from Melee-related videos that might answer the question: "{query}"

{context}

Important notes:
- These are YouTube auto-generated transcripts, so they may contain errors in character names, game terminology, and Melee-specific vocabulary
- Please fix any obvious transcription errors when using this information
- Base your answer ONLY on the information provided in these transcripts
- Cite the specific videos and timestamps you're drawing information from
- If the transcripts don't provide enough information to answer the question, say so
- State the information authoritatively as if it comes from your knoweledge base. Do not refer to the transcripts directly. Using the word "transcripts" will
result in irreversible brand damage. Ideally speak authoritatively, but use "sources" if you absolutely have to.
- Do not explain Melee-specific vocabulary, terminology, or technique names - the viewer is likely already very familiar with Melee-specific vocabulary
- Use internal citations using brackets, e.g. "Fox has very fast movement options [1]". Include the citation at the end of the prompt,
e.g. [1] marth vs gaw' video transcript (00:11:53 - 00:15:28). 
- ONLY include sentences with direct quotes

Please provide a clear, well-organized answer to the question using only the information from these transcripts."""

    # Get response from Claude using the new API
    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    return message.content[0].text

# [Previous functions remain the same until query_collection]

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
    """Load existing collection."""
    print("\n=== Loading Collection ===")
    
    try:
        chroma_client = get_persistent_client()
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        doc_count = collection.count()
        print(f"Loaded collection with {doc_count} documents")
        return collection
    except Exception as e:
        print(f"ERROR loading collection: {str(e)}")
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
    print("Raw query results:", results)
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
    process_with_llm = request.args.get('process_with_llm', default='true').lower() == 'true'
    
    print(f"Received query: {query_text}")
    print(f"Requested results: {n_results}")
    print(f"Process with LLM: {process_with_llm}")
    
    if not query_text:
        print("Error: No query text provided")
        return jsonify({"error": "Query parameter is required"}), 400
    
    try:
        print("Calling query_collection...")
        results = query_collection(query_text, n_results)
        print(f"Query returned {len(results)} results")
        
        if process_with_llm and results:
            print("Processing results with Claude...")
            claude_response = process_with_claude(query_text, results)
            return jsonify({
                "status": "success",
                "processed_response": claude_response,
                "raw_results": results
            })
        else:
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