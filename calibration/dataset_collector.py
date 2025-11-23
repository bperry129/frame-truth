#!/usr/bin/env python3
"""
FrameTruth Calibration Dataset Collector
Builds a labeled dataset for training evidence-based thresholds
"""

import os
import json
import csv
import requests
import yt_dlp
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import hashlib

@dataclass
class VideoMetadata:
    """Metadata for each video in calibration dataset"""
    video_id: str
    filename: str
    label: str  # 'real' or 'ai'
    source: str  # 'phone', 'sora', 'runway', etc.
    url: Optional[str] = None
    title: Optional[str] = None
    duration: Optional[float] = None
    resolution: Optional[str] = None
    fps: Optional[float] = None
    file_size: Optional[int] = None
    notes: Optional[str] = None

class CalibrationDatasetCollector:
    """Collects and organizes videos for calibration dataset"""
    
    def __init__(self, dataset_dir: str = "calibration/dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.dataset_dir / "real").mkdir(exist_ok=True)
        (self.dataset_dir / "ai").mkdir(exist_ok=True)
        (self.dataset_dir / "metadata").mkdir(exist_ok=True)
        
        self.metadata_file = self.dataset_dir / "metadata" / "dataset.csv"
        self.videos_metadata: List[VideoMetadata] = []
        
        # Load existing metadata if available
        self.load_existing_metadata()
    
    def load_existing_metadata(self):
        """Load existing dataset metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    metadata = VideoMetadata(**row)
                    self.videos_metadata.append(metadata)
            print(f"📊 Loaded {len(self.videos_metadata)} existing videos")
    
    def save_metadata(self):
        """Save dataset metadata to CSV"""
        with open(self.metadata_file, 'w', newline='') as f:
            if self.videos_metadata:
                fieldnames = list(asdict(self.videos_metadata[0]).keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for metadata in self.videos_metadata:
                    writer.writerow(asdict(metadata))
        print(f"💾 Saved metadata for {len(self.videos_metadata)} videos")
    
    def add_video_from_url(self, url: str, label: str, source: str, notes: str = ""):
        """Download and add video from URL"""
        try:
            # Generate unique ID from URL
            video_id = hashlib.md5(url.encode()).hexdigest()[:12]
            
            # Check if already exists
            if any(v.video_id == video_id for v in self.videos_metadata):
                print(f"⚠️ Video {video_id} already exists, skipping")
                return False
            
            # Download with yt-dlp
            output_dir = self.dataset_dir / label
            output_template = str(output_dir / f"{video_id}.%(ext)s")
            
            ydl_opts = {
                'format': 'best[height<=720]/best',
                'outtmpl': output_template,
                'quiet': True,
                'no_warnings': True,
                'max_filesize': 100 * 1024 * 1024,  # 100MB max
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                
                # Find the downloaded file
                ext = info.get('ext', 'mp4')
                filename = f"{video_id}.{ext}"
                filepath = output_dir / filename
                
                if not filepath.exists():
                    print(f"❌ Download failed for {url}")
                    return False
                
                # Create metadata
                metadata = VideoMetadata(
                    video_id=video_id,
                    filename=filename,
                    label=label,
                    source=source,
                    url=url,
                    title=info.get('title', ''),
                    duration=info.get('duration'),
                    resolution=f"{info.get('width', 0)}x{info.get('height', 0)}",
                    fps=info.get('fps'),
                    file_size=filepath.stat().st_size,
                    notes=notes
                )
                
                self.videos_metadata.append(metadata)
                print(f"✅ Added {label} video: {filename} ({source})")
                return True
                
        except Exception as e:
            print(f"❌ Error adding video from {url}: {e}")
            return False
    
    def add_local_video(self, filepath: str, label: str, source: str, notes: str = ""):
        """Add video from local file"""
        try:
            source_path = Path(filepath)
            if not source_path.exists():
                print(f"❌ File not found: {filepath}")
                return False
            
            # Generate unique ID from file path and size
            file_hash = hashlib.md5(f"{filepath}{source_path.stat().st_size}".encode()).hexdigest()[:12]
            video_id = f"local_{file_hash}"
            
            # Check if already exists
            if any(v.video_id == video_id for v in self.videos_metadata):
                print(f"⚠️ Video {video_id} already exists, skipping")
                return False
            
            # Copy to dataset directory
            output_dir = self.dataset_dir / label
            filename = f"{video_id}{source_path.suffix}"
            dest_path = output_dir / filename
            
            import shutil
            shutil.copy2(source_path, dest_path)
            
            # Create metadata
            metadata = VideoMetadata(
                video_id=video_id,
                filename=filename,
                label=label,
                source=source,
                url=None,
                title=source_path.stem,
                duration=None,  # Could extract with ffprobe
                resolution=None,
                fps=None,
                file_size=dest_path.stat().st_size,
                notes=notes
            )
            
            self.videos_metadata.append(metadata)
            print(f"✅ Added {label} video: {filename} ({source})")
            return True
            
        except Exception as e:
            print(f"❌ Error adding local video {filepath}: {e}")
            return False
    
    def get_dataset_stats(self) -> Dict:
        """Get statistics about the current dataset"""
        stats = {
            'total_videos': len(self.videos_metadata),
            'real_videos': len([v for v in self.videos_metadata if v.label == 'real']),
            'ai_videos': len([v for v in self.videos_metadata if v.label == 'ai']),
            'sources': {},
            'total_size_mb': 0
        }
        
        for video in self.videos_metadata:
            # Count by source
            if video.source not in stats['sources']:
                stats['sources'][video.source] = {'real': 0, 'ai': 0}
            stats['sources'][video.source][video.label] += 1
            
            # Total size
            if video.file_size:
                stats['total_size_mb'] += video.file_size / (1024 * 1024)
        
        stats['total_size_mb'] = round(stats['total_size_mb'], 1)
        return stats
    
    def print_dataset_summary(self):
        """Print a summary of the current dataset"""
        stats = self.get_dataset_stats()
        
        print("\n📊 CALIBRATION DATASET SUMMARY")
        print("=" * 50)
        print(f"Total Videos: {stats['total_videos']}")
        print(f"Real Videos: {stats['real_videos']}")
        print(f"AI Videos: {stats['ai_videos']}")
        print(f"Total Size: {stats['total_size_mb']} MB")
        print("\nBy Source:")
        for source, counts in stats['sources'].items():
            print(f"  {source}: {counts['real']} real, {counts['ai']} AI")
        print("=" * 50)

# Predefined video collections for quick dataset building
REAL_VIDEO_SOURCES = {
    "phone_recordings": [
        # Add URLs of known real phone recordings
        # "https://youtube.com/watch?v=...",
    ],
    "professional_cameras": [
        # Add URLs of known professional camera footage
    ],
    "webcam_recordings": [
        # Add URLs of known webcam footage
    ],
    "action_cameras": [
        # Add URLs of known GoPro/action camera footage
    ]
}

AI_VIDEO_SOURCES = {
    "sora": [
        # Add URLs of known Sora generations
        # "https://youtube.com/watch?v=...",
    ],
    "runway_gen3": [
        # Add URLs of known Runway Gen-3 videos
    ],
    "pika_labs": [
        # Add URLs of known Pika Labs videos
    ],
    "kling_ai": [
        # Add URLs of known Kling AI videos
    ],
    "luma_dream": [
        # Add URLs of known Luma Dream Machine videos
    ]
}

def main():
    """Main function for interactive dataset collection"""
    collector = CalibrationDatasetCollector()
    
    print("🎯 FrameTruth Calibration Dataset Collector")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Add video from URL")
        print("2. Add local video file")
        print("3. Batch add from predefined sources")
        print("4. Show dataset summary")
        print("5. Save and exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            url = input("Video URL: ").strip()
            label = input("Label (real/ai): ").strip().lower()
            source = input("Source (phone/sora/runway/etc): ").strip()
            notes = input("Notes (optional): ").strip()
            
            if label in ['real', 'ai']:
                collector.add_video_from_url(url, label, source, notes)
            else:
                print("❌ Label must be 'real' or 'ai'")
        
        elif choice == "2":
            filepath = input("Local file path: ").strip()
            label = input("Label (real/ai): ").strip().lower()
            source = input("Source (phone/sora/runway/etc): ").strip()
            notes = input("Notes (optional): ").strip()
            
            if label in ['real', 'ai']:
                collector.add_local_video(filepath, label, source, notes)
            else:
                print("❌ Label must be 'real' or 'ai'")
        
        elif choice == "3":
            print("🚧 Batch adding not implemented yet")
            print("Please add URLs to the REAL_VIDEO_SOURCES and AI_VIDEO_SOURCES dictionaries")
        
        elif choice == "4":
            collector.print_dataset_summary()
        
        elif choice == "5":
            collector.save_metadata()
            print("👋 Dataset saved. Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()
