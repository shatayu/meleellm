import chromadb
import pickle
from typing import List, Dict
import uuid
import json
import os
from chromadb.config import Settings
from chromadb.errors import InvalidCollectionException

def get_persistent_client(persist_dir: str = "chroma_db"):
    """Initialize ChromaDB client with persistence."""
    return chromadb.PersistentClient(path=persist_dir)

def load_processed_videos(pickle_file: str) -> List[Dict]:
    """Load processed video chunks from pickle file."""
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)

def create_or_load_collection(chunks: List[Dict], collection_name: str = "video_transcripts", persist_dir: str = "chroma_db"):
    """Create a new collection or load existing one with data verification."""
    # Initialize persistent client
    chroma_client = get_persistent_client(persist_dir)
    
    # Try to get existing collection
    collection = None
    try:
        collection = chroma_client.get_collection(name=collection_name)
        print(f"Found existing collection '{collection_name}'")
        
        # Verify if collection needs updating
        existing_count = collection.count()
        if existing_count == len(chunks):
            print(f"Collection is up to date with {existing_count} chunks")
            return collection
        
        print(f"Collection size mismatch: {existing_count} vs {len(chunks)} chunks")
        print("Recreating collection with updated data...")
        chroma_client.delete_collection(name=collection_name)
        
    except InvalidCollectionException:
        print(f"Collection '{collection_name}' does not exist. Creating new collection...")
    
    # Create new collection
    collection = chroma_client.create_collection(name=collection_name)
    
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
        print(f"Added chunks {i} to {end_idx}")
    
    return collection

def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def query_collection(collection, query_text: str, n_results: int = 3):
    """Query the collection and return formatted results."""
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

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Create and query video transcript vector database')
    parser.add_argument('pickle_file', help='Pickle file containing processed video chunks')
    parser.add_argument('--collection', default='video_transcripts',
                       help='Name of ChromaDB collection (default: video_transcripts)')
    parser.add_argument('--persist-dir', default='chroma_db',
                       help='Directory for persistent storage (default: chroma_db)')
    
    args = parser.parse_args()
    
    # Ensure persist directory exists
    os.makedirs(args.persist_dir, exist_ok=True)
    
    # Load chunks from pickle file
    print(f"Loading chunks from {args.pickle_file}")
    chunks = load_processed_videos(args.pickle_file)
    print(f"Loaded {len(chunks)} chunks")
    
    # Create or load collection
    collection = create_or_load_collection(
        chunks, 
        args.collection,
        args.persist_dir
    )
    print(f"Collection '{args.collection}' ready for queries")
    
    # Interactive query loop
    print("\nEnter queries (or 'quit' to exit):")
    while True:
        query = input("\nQuery: ").strip()
        if query.lower() == 'quit':
            break
            
        results = query_collection(collection, query)
        print("\nResults:")
        for result in results:
            print(f"\nFrom video: {result['video_title']}")
            print(f"At timestamp: {result['timestamp']}")
            print(f"URL: {result['video_url']}")
            print(f"Text: {result['text']}")
            print("-" * 80)

if __name__ == "__main__":
    main()