import chromadb
import pickle
import uuid
import os

print(f"Using ChromaDB version: {chromadb.__version__}")

def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def build_database():
    # Remove existing database if it exists
    if os.path.exists("chroma_db"):
        print("Removing existing database...")
        import shutil
        shutil.rmtree("chroma_db")
    
    # Create the database directory
    os.makedirs("chroma_db", exist_ok=True)
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path="chroma_db")
    
    # Load chunks
    print("Loading pickle file...")
    with open("processed_videos.pkl", "rb") as f:
        chunks = pickle.load(f)
    print(f"Loaded {len(chunks)} chunks")
    
    # Create collection
    print("Creating collection...")
    collection = client.create_collection(name="video_transcripts")
    
    # Process in batches
    batch_size = 1000
    for i in range(0, len(chunks), batch_size):
        end_idx = min(i + batch_size, len(chunks))
        print(f"Processing batch {i} to {end_idx}...")
        
        batch_documents = []
        batch_metadatas = []
        batch_ids = []
        
        for chunk in chunks[i:end_idx]:
            chunk_id = str(uuid.uuid4())
            batch_documents.append(chunk['text'])
            
            metadata = {
                'video_title': chunk['video_title'],
                'video_url': chunk['video_url'],
                'video_id': chunk['video_id'],
                'start_time': chunk['start_time'],
                'end_time': chunk['end_time'],
                'timestamp': f"{format_timestamp(chunk['start_time'])} - {format_timestamp(chunk['end_time'])}"
            }
            
            batch_metadatas.append(metadata)
            batch_ids.append(chunk_id)
        
        collection.add(
            documents=batch_documents,
            metadatas=batch_metadatas,
            ids=batch_ids
        )
        print(f"Added batch {i} to {end_idx}")
    
    print("Database build complete!")
    print(f"Total documents in collection: {collection.count()}")
    
    # Verify the collection is queryable
    print("\nTesting a query...")
    results = collection.query(
        query_texts=["test query"],
        n_results=1
    )
    print(f"Query test successful, got {len(results['documents'][0])} results")

if __name__ == "__main__":
    build_database()