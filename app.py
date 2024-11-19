import boto3
import zipfile
import shutil
from pathlib import Path
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
import botocore.exceptions

app = Flask(__name__)
CORS(app)


# Configuration
# Add these with your existing config section
AWS_ACCESS_KEY_ID=os.getenv('AWS_ACCESS_KEY_ID', 'none')
AWS_SECRET_ACCESS_KEY=os.getenv('AWS_SECRET_ACCESS_KEY', 'none')
AWS_BUCKET_NAME = os.getenv('AWS_BUCKET_NAME', 'meleellm-vectordb')
AWS_OBJECT_KEY = os.getenv('AWS_OBJECT_KEY', 'chroma_db.zip')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
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
e.g. [1] some video. 
- ONLY include sentences with direct quotes


Please provide a clear, well-organized answer to the question using only the information from these transcripts."""

    # Get response from Claude using the new API
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
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

def download_and_prepare_db():
    """Download ChromaDB from S3 and prepare it for use."""
    print("\n=== Downloading ChromaDB from S3 ===")
    
    # Create a temporary directory for the zip file
    zip_path = Path('temp_chroma.zip')
    
    try:
        # Check for AWS credentials
        if not os.getenv('AWS_ACCESS_KEY_ID') or not os.getenv('AWS_SECRET_ACCESS_KEY'):
            raise ValueError("AWS credentials not found in environment variables. "
                           "Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")

        # Initialize S3 client
        try:
            s3 = boto3.client('s3', region_name=AWS_REGION)
            # Test connection by checking if bucket exists
            s3.head_bucket(Bucket=AWS_BUCKET_NAME)
        except boto3.exceptions.NoCredentialsError:
            raise ValueError("AWS credentials are invalid or not properly configured.")
        except botocore.exceptions.ClientError as e:  # Changed this line
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == '403':
                raise ValueError(f"Access denied to bucket {AWS_BUCKET_NAME}. Check your AWS permissions.")
            elif error_code == '404':
                raise ValueError(f"Bucket {AWS_BUCKET_NAME} does not exist.")
            else:
                raise ValueError(f"AWS Error: {str(e)}")
        
        print(f"Downloading from s3://{AWS_BUCKET_NAME}/{AWS_OBJECT_KEY}")
        
        # Download the zip file
        try:
            s3.download_file(AWS_BUCKET_NAME, AWS_OBJECT_KEY, str(zip_path))
            print("Successfully downloaded database zip")
        except botocore.exceptions.ClientError as e:  # Changed this line
            if e.response.get('Error', {}).get('Code', '') == '404':
                raise ValueError(f"File {AWS_OBJECT_KEY} not found in bucket {AWS_BUCKET_NAME}")
            else:
                raise ValueError(f"Error downloading file: {str(e)}")

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

# Add this at the top level, after your app declaration but before the routes
print("\n=== Application Starting ===")

try:
    # Download and prepare database
    download_and_prepare_db()
    
    # Initialize collection on startup
    print("Initializing collection on startup...")
    collection = create_or_load_collection()
    print("Successfully initialized collection on startup")
except Exception as e:
    print(f"Failed to initialize application: {str(e)}")
    print(f"Full traceback: {traceback.format_exc()}")
    raise

# Keep this part for local development
if __name__ == '__main__':
    port = int(os.getenv('PORT', 10000))
    print(f"\nStarting Flask application on port {port}")
    app.run(host='0.0.0.0', port=port)