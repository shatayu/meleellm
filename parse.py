import asyncio
import re
from typing import List, Dict, Optional
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time
from datetime import datetime
import sys
import argparse
import pickle

@dataclass
class VideoMetadata:
    title: str
    url: str
    video_id: str
    duration: int
    upload_date: str

def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?]*)',
        r'(?:youtube\.com\/embed\/)([^&\n?]*)',
        r'(?:youtube\.com\/v\/)([^&\n?]*)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {url}")

def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def get_video_metadata(url: str) -> Optional[VideoMetadata]:
    """Get video metadata using yt-dlp."""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return VideoMetadata(
                title=info.get('title', 'Unknown Title'),
                url=url,
                video_id=info.get('id', 'Unknown ID'),
                duration=info.get('duration', 0),
                upload_date=info.get('upload_date', 'Unknown Date')
            )
    except Exception as e:
        print(f"Error getting metadata for {url}: {e}")
        return None

def process_single_video(url: str, chunk_size: int = 300, overlap: int = 100) -> Optional[Dict]:
    """Process video with overlapping chunks for better context."""
    try:
        metadata = get_video_metadata(url)
        if not metadata:
            return None
            
        video_id = metadata.video_id
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        
        # Combine transcript text with timestamps
        chunks = []
        current_chunk = []
        current_text = ""
        chunk_start = None
        
        for entry in transcript:
            if not chunk_start:
                chunk_start = entry['start']
                
            current_text += " " + entry['text']
            current_chunk.append(entry)
            
            # If we've reached desired chunk size, save it
            if len(current_text.split()) >= chunk_size:
                chunks.append({
                    'text': current_text.strip(),
                    'start_time': chunk_start,
                    'end_time': entry['start'] + entry['duration'],
                    'video_title': metadata.title,
                    'video_url': url,
                    'video_id': video_id
                })
                
                # Keep overlap portion for next chunk
                overlap_entries = current_chunk[-overlap:]
                current_chunk = overlap_entries
                current_text = " ".join(e['text'] for e in overlap_entries)
                chunk_start = overlap_entries[0]['start']
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append({
                'text': current_text.strip(),
                'start_time': chunk_start,
                'end_time': current_chunk[-1]['start'] + current_chunk[-1]['duration'],
                'video_title': metadata.title,
                'video_url': url,
                'video_id': video_id
            })
        
        return {
            'metadata': {
                'title': metadata.title,
                'url': url,
                'video_id': video_id,
                'duration': metadata.duration,
                'upload_date': metadata.upload_date,
                'processed_date': datetime.now().isoformat()
            },
            'chunks': chunks
        }
        
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None

async def process_videos(urls: List[str], max_concurrent: int = 5) -> List[Dict]:
    """Process multiple videos concurrently with rate limiting."""
    results = []
    
    # Create a thread pool for concurrent processing
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        loop = asyncio.get_event_loop()
        
        # Create tasks for each URL
        tasks = []
        for url in urls:
            task = loop.run_in_executor(executor, process_single_video, url)
            tasks.append(task)
        
        # Wait for all tasks to complete
        completed_results = await asyncio.gather(*tasks)
        
        # Filter out None results (failed processing)
        results = [r for r in completed_results if r is not None]
    
    return results

def read_urls_from_file(filename: str) -> List[str]:
    """
    Read URLs from a text file, ignoring comments and empty lines.
    Comments start with #.
    """
    urls = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            # Strip whitespace
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
                
            urls.append(line)
    
    return urls

def save_processed_videos(results: List[Dict], output_file: str = 'processed_videos.pkl'):
    """Save processed videos in format ready for vector DB."""
    # Flatten all chunks from all videos into single list
    all_chunks = []
    for result in results:
        if result and 'chunks' in result:
            all_chunks.extend(result['chunks'])
    
    with open(output_file, 'wb') as f:
        pickle.dump(all_chunks, f)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process YouTube video transcripts from a file.')
    parser.add_argument('filename', help='Text file containing YouTube URLs (one per line)')
    parser.add_argument('--max-concurrent', type=int, default=5, 
                       help='Maximum number of videos to process concurrently (default: 5)')
    parser.add_argument('--output', default='processed_videos.pkl',
                       help='Output pickle file (default: processed_videos.pkl)')
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Read URLs from file
        print(f"Reading URLs from {args.filename}")
        urls = read_urls_from_file(args.filename)
        print(f"Found {len(urls)} URLs to process")
        
        if not urls:
            print("No valid URLs found in file")
            return
        
        # Process videos
        results = asyncio.run(process_videos(urls, max_concurrent=args.max_concurrent))
        
        # Save results
        save_processed_videos(results, args.output)
        
        # Print summary
        print(f"\nProcessed {len(results)} videos successfully")
        print(f"Saved chunks to {args.output}")
        
        # Print some stats
        total_chunks = sum(len(result['chunks']) for result in results if result and 'chunks' in result)
        print(f"Total chunks created: {total_chunks}")
        
        for result in results:
            print(f"\nTitle: {result['metadata']['title']}")
            print(f"Duration: {result['metadata']['duration']} seconds")
            print(f"Number of chunks: {len(result['chunks'])}")
            
    except FileNotFoundError:
        print(f"Error: File '{args.filename}' not found")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()