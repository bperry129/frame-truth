#!/usr/bin/env python3
"""
🚀 Research Dataset Downloader for Frame Truth
Automatically downloads and integrates research-grade AI video detection datasets

Supported Datasets:
1. Synth-Vid-Detect (HuggingFace: ductai199x/synth-vid-detect)
   - Contains: Sora, Pika, CogVideo, VideoCrafter, Stable Video Diffusion + real videos
   - Pre-labeled with generator type
   - Thousands of videos

2. GenVidBench (if available)
   - Multi-generator benchmark for AI video detection
   - Research-grade evaluation dataset

Usage:
    python research_dataset_downloader.py --dataset synth-vid-detect --max-videos 500
    python research_dataset_downloader.py --dataset all --max-videos 1000
"""

import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from tqdm import tqdm
import hashlib
import shutil

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import DATA_DIR, RAPIDAPI_KEY
from download_videos import generate_video_id

try:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, list_repo_files
    HF_AVAILABLE = True
except ImportError:
    print("⚠️ HuggingFace datasets not installed. Run: pip install datasets huggingface-hub")
    HF_AVAILABLE = False

class ResearchDatasetDownloader:
    """Downloads and integrates research datasets for AI video detection"""
    
    def __init__(self, max_videos_per_dataset: int = 500):
        self.max_videos = max_videos_per_dataset
        self.data_dir = Path(DATA_DIR)
        self.research_dir = self.data_dir / "research_datasets"
        self.research_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.research_dir / "raw_videos").mkdir(exist_ok=True)
        (self.research_dir / "metadata").mkdir(exist_ok=True)
        
        # Generator mapping for consistent labeling
        self.generator_mapping = {
            # AI Generators
            'sora': 'ai',
            'pika': 'ai', 
            'cogvideo': 'ai',
            'videocrafter': 'ai',
            'stable_video_diffusion': 'ai',
            'stable-video-diffusion': 'ai',
            'svd': 'ai',
            'runway': 'ai',
            'runway_gen3': 'ai',
            'runway-gen3': 'ai',
            'gen3': 'ai',
            'kling': 'ai',
            'luma': 'ai',
            'haiper': 'ai',
            'midjourney': 'ai',
            'synthesia': 'ai',
            'artificial': 'ai',
            'generated': 'ai',
            'synthetic': 'ai',
            'ai': 'ai',
            
            # Real videos
            'real': 'real',
            'authentic': 'real',
            'camera': 'real',
            'natural': 'real',
            'human': 'real'
        }
    
    def download_synth_vid_detect(self) -> Dict:
        """
        Download Synth-Vid-Detect dataset from HuggingFace
        Dataset: ductai199x/synth-vid-detect
        """
        if not HF_AVAILABLE:
            return {"error": "HuggingFace datasets not available"}
        
        print("🔬 Downloading Synth-Vid-Detect dataset from HuggingFace...")
        print("   Dataset: ductai199x/synth-vid-detect")
        print("   Contains: Sora, Pika, CogVideo, VideoCrafter, SVD + real videos")
        
        try:
            # Try different approaches to load the dataset
            print("   Attempting method 1: Direct dataset loading...")
            try:
                dataset = load_dataset("ductai199x/synth-vid-detect", split="train", streaming=True)
            except Exception as e1:
                print(f"   Method 1 failed: {e1}")
                print("   Attempting method 2: Without custom script...")
                try:
                    dataset = load_dataset("ductai199x/synth-vid-detect", split="train", streaming=True, trust_remote_code=False)
                except Exception as e2:
                    print(f"   Method 2 failed: {e2}")
                    print("   Attempting method 3: Manual file access...")
                    # Try to access files directly from the repo
                    from huggingface_hub import list_repo_files
                    files = list_repo_files("ductai199x/synth-vid-detect")
                    print(f"   Available files: {files[:10]}...")  # Show first 10 files
                    
                    # Look for data files
                    data_files = [f for f in files if f.endswith(('.csv', '.json', '.parquet', '.jsonl'))]
                    if data_files:
                        print(f"   Found data files: {data_files}")
                        # Try loading with specific data files
                        dataset = load_dataset("ductai199x/synth-vid-detect", data_files=data_files[0], split="train", streaming=True)
                    else:
                        raise Exception("No accessible data files found in repository")
            
            downloaded_videos = []
            failed_downloads = []
            skipped_videos = []
            
            # Process videos with progress bar
            video_count = 0
            for item in tqdm(dataset, desc="Processing Synth-Vid-Detect", total=self.max_videos):
                if video_count >= self.max_videos:
                    break
                
                try:
                    # Extract video information
                    video_url = item.get('video_url') or item.get('url') or item.get('video')
                    label_raw = item.get('label') or item.get('generator') or item.get('source', 'unknown')
                    
                    if not video_url:
                        print(f"⚠️ No video URL found in item: {item.keys()}")
                        continue
                    
                    # Map label to our format
                    label_clean = str(label_raw).lower().strip()
                    label = self.generator_mapping.get(label_clean, 'unknown')
                    
                    if label == 'unknown':
                        # Try to infer from label content
                        if any(ai_term in label_clean for ai_term in ['ai', 'generated', 'synthetic', 'sora', 'pika']):
                            label = 'ai'
                        elif any(real_term in label_clean for real_term in ['real', 'authentic', 'camera']):
                            label = 'real'
                        else:
                            print(f"⚠️ Unknown label: {label_raw}, skipping...")
                            continue
                    
                    # Generate consistent video ID
                    video_id = generate_video_id(video_url)
                    
                    # Check if already downloaded
                    existing_files = list((self.research_dir / "raw_videos").glob(f"{video_id}.*"))
                    if existing_files:
                        skipped_videos.append({
                            "video_id": video_id,
                            "url": video_url,
                            "label": label,
                            "generator": label_raw,
                            "reason": "already_exists"
                        })
                        video_count += 1
                        continue
                    
                    # Download video
                    result = self.download_research_video(video_url, video_id, label, str(label_raw))
                    
                    if result["success"]:
                        downloaded_videos.append(result)
                        print(f"✅ Downloaded {video_id}: {label_raw} -> {label}")
                    else:
                        failed_downloads.append({
                            "video_id": video_id,
                            "url": video_url,
                            "label": label,
                            "generator": label_raw,
                            "error": result["error"]
                        })
                        print(f"❌ Failed {video_id}: {result['error']}")
                    
                    video_count += 1
                    
                except Exception as e:
                    print(f"❌ Error processing item: {e}")
                    continue
            
            # Save metadata
            metadata = {
                "dataset": "synth-vid-detect",
                "source": "ductai199x/synth-vid-detect",
                "total_processed": video_count,
                "downloaded": len(downloaded_videos),
                "failed": len(failed_downloads),
                "skipped": len(skipped_videos),
                "videos": downloaded_videos,
                "failed_videos": failed_downloads,
                "skipped_videos": skipped_videos
            }
            
            metadata_path = self.research_dir / "metadata" / "synth_vid_detect.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Synth-Vid-Detect download complete:")
            print(f"   Downloaded: {len(downloaded_videos)} videos")
            print(f"   Failed: {len(failed_downloads)} videos")
            print(f"   Skipped: {len(skipped_videos)} videos")
            
            return metadata
            
        except Exception as e:
            error_msg = f"Failed to download Synth-Vid-Detect: {str(e)}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}
    
    def download_research_video(self, url: str, video_id: str, label: str, generator: str) -> Dict:
        """Download a single video from research dataset"""
        try:
            # Use the same download logic as the main system
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': '*/*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
            }
            
            # Try direct download first
            response = requests.get(url, stream=True, timeout=60, headers=headers)
            
            if response.status_code != 200:
                return {"success": False, "error": f"HTTP {response.status_code}"}
            
            # Determine file extension
            content_type = response.headers.get('content-type', '')
            if 'mp4' in content_type:
                ext = 'mp4'
            elif 'webm' in content_type:
                ext = 'webm'
            elif 'avi' in content_type:
                ext = 'avi'
            else:
                ext = 'mp4'  # Default
            
            # Save video file
            filepath = self.research_dir / "raw_videos" / f"{video_id}.{ext}"
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Verify file was created and has content
            if not filepath.exists() or filepath.stat().st_size == 0:
                return {"success": False, "error": "Downloaded file is empty"}
            
            return {
                "success": True,
                "video_id": video_id,
                "filepath": str(filepath),
                "label": label,
                "generator": generator,
                "url": url,
                "file_size": filepath.stat().st_size
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def explore_genvidbench(self) -> Dict:
        """
        Explore GenVidBench dataset availability
        This is a research dataset that may not be publicly available yet
        """
        print("🔍 Exploring GenVidBench dataset availability...")
        
        # Try common HuggingFace dataset names for GenVidBench
        potential_names = [
            "genvidbench/genvidbench",
            "genvidbench/dataset", 
            "ai-video-detection/genvidbench",
            "video-detection/genvidbench"
        ]
        
        for dataset_name in potential_names:
            try:
                print(f"   Trying: {dataset_name}")
                dataset = load_dataset(dataset_name, split="train", streaming=True)
                print(f"✅ Found GenVidBench at: {dataset_name}")
                return {"available": True, "name": dataset_name}
            except Exception as e:
                print(f"   ❌ Not found: {e}")
                continue
        
        print("⚠️ GenVidBench not found on HuggingFace")
        print("   This dataset may not be publicly available yet")
        return {"available": False, "error": "Dataset not found"}
    
    def create_unified_csv(self) -> str:
        """
        Create a unified CSV file combining all research datasets
        Compatible with existing training pipeline
        """
        print("📊 Creating unified research dataset CSV...")
        
        all_videos = []
        
        # Load Synth-Vid-Detect metadata
        synth_metadata_path = self.research_dir / "metadata" / "synth_vid_detect.json"
        if synth_metadata_path.exists():
            with open(synth_metadata_path, 'r') as f:
                synth_data = json.load(f)
            
            for video in synth_data.get("videos", []):
                all_videos.append({
                    "url": video["url"],
                    "label": video["label"],
                    "generator": video["generator"],
                    "dataset": "synth-vid-detect",
                    "video_id": video["video_id"],
                    "filepath": video["filepath"]
                })
        
        # Create DataFrame
        df = pd.DataFrame(all_videos)
        
        # Save unified CSV
        csv_path = self.research_dir / "research_dataset.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"✅ Created unified CSV: {csv_path}")
        print(f"   Total videos: {len(df)}")
        print(f"   AI videos: {len(df[df['label'] == 'ai'])}")
        print(f"   Real videos: {len(df[df['label'] == 'real'])}")
        
        return str(csv_path)
    
    def download_all_datasets(self) -> Dict:
        """Download all available research datasets"""
        print("🚀 Starting automated research dataset download...")
        
        results = {
            "synth_vid_detect": None,
            "genvidbench": None,
            "total_videos": 0,
            "unified_csv": None
        }
        
        # Download Synth-Vid-Detect
        synth_result = self.download_synth_vid_detect()
        results["synth_vid_detect"] = synth_result
        
        if "error" not in synth_result:
            results["total_videos"] += synth_result.get("downloaded", 0)
        
        # Explore GenVidBench
        genvidbench_result = self.explore_genvidbench()
        results["genvidbench"] = genvidbench_result
        
        # Create unified CSV
        if results["total_videos"] > 0:
            csv_path = self.create_unified_csv()
            results["unified_csv"] = csv_path
        
        print(f"\n🎉 Research dataset download complete!")
        print(f"   Total videos downloaded: {results['total_videos']}")
        print(f"   Unified CSV: {results['unified_csv']}")
        
        return results

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Download research datasets for AI video detection")
    parser.add_argument("--dataset", choices=["synth-vid-detect", "genvidbench", "all"], 
                       default="all", help="Which dataset to download")
    parser.add_argument("--max-videos", type=int, default=500, 
                       help="Maximum videos to download per dataset")
    
    args = parser.parse_args()
    
    if not HF_AVAILABLE:
        print("❌ HuggingFace datasets not installed!")
        print("   Run: pip install datasets huggingface-hub")
        return
    
    downloader = ResearchDatasetDownloader(max_videos_per_dataset=args.max_videos)
    
    if args.dataset == "synth-vid-detect":
        result = downloader.download_synth_vid_detect()
    elif args.dataset == "genvidbench":
        result = downloader.explore_genvidbench()
    else:  # all
        result = downloader.download_all_datasets()
    
    print(f"\n📊 Final Result:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
