"""
Video Download Module for FrameTruth Training Pipeline
Uses the same RapidAPI Social Download All-in-One API as the main site
"""

import requests
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional
import time

from config import RAPIDAPI_KEY, RAPIDAPI_ENDPOINT, DATA_DIR

def generate_video_id(url: str) -> str:
    """Generate unique video ID from URL"""
    return hashlib.md5(url.encode()).hexdigest()[:12]

def download_video(url: str, video_id: str) -> Dict:
    """
    Download video using RapidAPI Social Download All-in-One
    Same logic as main site's unified API downloader
    """
    print(f"🔄 Downloading video {video_id} from: {url}")
    
    try:
        # Correct headers for Social Download All in One API
        headers = {
            'x-rapidapi-host': 'social-download-all-in-one.p.rapidapi.com',
            'x-rapidapi-key': RAPIDAPI_KEY,
            'Content-Type': 'application/json'
        }
        
        # API payload
        payload = {
            "url": url
        }
        
        print(f"📤 Making request to: {RAPIDAPI_ENDPOINT}")
        
        response = requests.post(
            RAPIDAPI_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"📥 Response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ API failed with status {response.status_code}")
            print(f"❌ Response text: {response.text[:500]}")
            raise Exception(f"API returned status {response.status_code}: {response.text[:200]}")
        
        data = response.json()
        print(f"📥 API response keys: {list(data.keys())}")
        
        # Extract download URL from the documented response format
        download_url = None
        video_info = {}
        
        # Handle the documented API response format
        if data.get('medias') and isinstance(data['medias'], list) and len(data['medias']) > 0:
            # Find the best video quality
            video_media = None
            for media in data['medias']:
                if media.get('type') == 'video':
                    video_media = media
                    # Prefer higher quality if available
                    if 'quality' in media and ('720' in str(media['quality']) or 'high' in str(media['quality']).lower()):
                        break
            
            if not video_media:
                # If no video type found, use first media item
                video_media = data['medias'][0]
            
            download_url = video_media.get('url')
            
            # Extract metadata from main response
            video_info = {
                "title": data.get("title", "Video"),
                "uploader": data.get("author", "Unknown"),
                "duration": data.get("duration"),
                "source": data.get("source", "unified_api"),
                "quality": video_media.get("quality", "unknown"),
                "extension": video_media.get("extension", "mp4")
            }
            
        elif data.get('url'):
            # Direct URL response (fallback)
            download_url = data['url']
            video_info = {
                "title": data.get("title", "Video"),
                "uploader": data.get("author", "Unknown"),
                "source": data.get("source", "unified_api")
            }
        
        if not download_url:
            raise Exception(f"No download URL found in API response. Response structure: {list(data.keys())}")
        
        print(f"📥 Downloading from: {download_url[:100]}...")
        
        # Download the video with proper headers
        video_response = requests.get(download_url, stream=True, timeout=120, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://download-all-in-one-elite.p.rapidapi.com/',
            'Accept': '*/*'
        })
        video_response.raise_for_status()
        
        # Determine file extension
        extension = video_info.get('extension', 'mp4')
        if not extension.startswith('.'):
            extension = f".{extension}"
        
        filename = f"{video_id}{extension}"
        filepath = DATA_DIR / "raw_videos" / filename
        
        # Save video file
        with open(filepath, 'wb') as f:
            total_size = 0
            for chunk in video_response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
        
        print(f"✅ Download successful: {filename} ({total_size / 1024 / 1024:.1f} MB)")
        
        return {
            "success": True,
            "filename": filename,
            "filepath": str(filepath),
            "size_mb": total_size / 1024 / 1024,
            "metadata": video_info
        }
        
    except Exception as e:
        print(f"❌ Download failed for {video_id}: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "video_id": video_id,
            "url": url
        }

def download_from_csv(csv_path: str) -> Dict:
    """
    Download all videos from a CSV file
    Expected format: url,label
    """
    import pandas as pd
    
    print(f"📊 Loading video list from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"📋 Found {len(df)} videos to download")
        
        if 'url' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV must have 'url' and 'label' columns")
        
        results = {
            "successful": [],
            "failed": [],
            "skipped": []
        }
        
        for idx, row in df.iterrows():
            url = row['url']
            label = row['label']
            video_id = generate_video_id(url)
            
            print(f"\n📹 Processing {idx+1}/{len(df)}: {video_id} ({label})")
            
            # Check if already downloaded
            existing_files = list((DATA_DIR / "raw_videos").glob(f"{video_id}.*"))
            if existing_files:
                print(f"⚠️ Video {video_id} already exists, skipping download")
                results["skipped"].append({
                    "video_id": video_id,
                    "url": url,
                    "label": label,
                    "existing_file": str(existing_files[0])
                })
                continue
            
            # Download video
            result = download_video(url, video_id)
            
            if result["success"]:
                result.update({
                    "video_id": video_id,
                    "url": url,
                    "label": label
                })
                results["successful"].append(result)
            else:
                results["failed"].append(result)
            
            # Small delay to be nice to the API
            time.sleep(1)
        
        print(f"\n📊 DOWNLOAD SUMMARY:")
        print(f"✅ Successful: {len(results['successful'])}")
        print(f"⚠️ Skipped: {len(results['skipped'])}")
        print(f"❌ Failed: {len(results['failed'])}")
        
        return results
        
    except Exception as e:
        print(f"❌ Failed to process CSV: {str(e)}")
        return {"error": str(e)}

def main():
    """Test the downloader with a sample URL"""
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll for testing
    video_id = generate_video_id(test_url)
    
    print("🧪 Testing video downloader...")
    result = download_video(test_url, video_id)
    
    if result["success"]:
        print(f"✅ Test successful! Downloaded: {result['filename']}")
    else:
        print(f"❌ Test failed: {result['error']}")

if __name__ == "__main__":
    main()
