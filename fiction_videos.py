import yt_dlp
import csv
from typing import List, Dict
import re

def get_channel_videos(channel_url: str, search_term: str) -> List[Dict]:
    """
    Get all videos from a channel that contain the search term in their title.
    Returns a list of dictionaries with video information.
    """
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'force_generic_extractor': False,
        'ignoreerrors': True
    }

    matching_videos = []
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract channel info
            channel_info = ydl.extract_info(channel_url, download=False)
            
            if 'entries' in channel_info:
                # Process each video
                for video in channel_info['entries']:
                    if video is None:
                        continue
                        
                    title = video.get('title', '').lower()
                    if search_term.lower() in title:
                        matching_videos.append({
                            'title': video['title'],
                            'url': f"https://www.youtube.com/watch?v={video['id']}"
                        })
    
    except Exception as e:
        print(f"Error: {e}")
        
    return matching_videos

def save_to_csv(videos: List[Dict], filename: str):
    """Save videos to CSV file with title and URL."""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['title', 'url'])
        writer.writeheader()
        writer.writerows(videos)

def save_to_txt(videos: List[Dict], filename: str):
    """Save just the URLs to a text file."""
    with open(filename, 'w', encoding='utf-8') as f:
        for video in videos:
            f.write(f"{video['url']}\n")

def main():
    channel_url = "https://www.youtube.com/@Fiction52/videos"
    search_term = "lesson"
    
    print(f"Fetching videos from channel that contain '{search_term}'...")
    videos = get_channel_videos(channel_url, search_term)
    
    if not videos:
        print("No matching videos found.")
        return
        
    print(f"\nFound {len(videos)} matching videos!")
    
    # Save to CSV
    csv_filename = "lesson_videos.csv"
    save_to_csv(videos, csv_filename)
    print(f"Saved video details to {csv_filename}")
    
    # Save to TXT
    txt_filename = "lesson_videos.txt"
    save_to_txt(videos, txt_filename)
    print(f"Saved video URLs to {txt_filename}")
    
    # Print summary
    print("\nMatching videos:")
    for video in videos:
        print(f"- {video['title']}")

if __name__ == "__main__":
    main()