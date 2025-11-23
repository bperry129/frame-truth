"""
Frame Extraction Module for FrameTruth Training Pipeline
Extracts frames evenly spaced across video duration
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import json

from config import DATA_DIR, FRAMES_PER_VIDEO, FRAME_SIZE

def extract_frames(video_path: str, video_id: str, num_frames: int = FRAMES_PER_VIDEO) -> Dict:
    """
    Extract frames evenly spaced across video duration
    
    Args:
        video_path: Path to video file
        video_id: Unique video identifier
        num_frames: Number of frames to extract
    
    Returns:
        Dict with frame paths and metadata
    """
    print(f"🎬 Extracting {num_frames} frames from video {video_id}")
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Video has no frames: {video_path}")
        
        print(f"📊 Video info: {total_frames} frames, {fps:.1f} fps, {width}x{height}, {duration:.1f}s")
        
        # Create output directory for this video's frames
        frames_dir = DATA_DIR / "frames" / video_id
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate frame indices to extract (evenly spaced)
        if num_frames >= total_frames:
            # Extract all frames if video is very short
            frame_indices = list(range(total_frames))
        else:
            # Evenly spaced frames
            step = total_frames / num_frames
            frame_indices = [int(i * step) for i in range(num_frames)]
        
        extracted_frames = []
        successful_extractions = 0
        
        for i, frame_idx in enumerate(frame_indices):
            # Seek to specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                print(f"⚠️ Failed to read frame {frame_idx}")
                continue
            
            # Resize frame to standard size for processing
            if FRAME_SIZE:
                frame_resized = cv2.resize(frame, FRAME_SIZE)
            else:
                frame_resized = frame
            
            # Save frame
            frame_filename = f"frame_{i:04d}.jpg"
            frame_path = frames_dir / frame_filename
            
            # Save with high quality
            cv2.imwrite(str(frame_path), frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            extracted_frames.append({
                "frame_index": i,
                "original_frame_number": frame_idx,
                "timestamp": frame_idx / fps if fps > 0 else 0,
                "filename": frame_filename,
                "path": str(frame_path)
            })
            
            successful_extractions += 1
        
        cap.release()
        
        if successful_extractions == 0:
            raise ValueError(f"No frames could be extracted from {video_path}")
        
        # Save frame metadata
        metadata = {
            "video_id": video_id,
            "video_path": video_path,
            "total_frames": total_frames,
            "fps": fps,
            "duration": duration,
            "original_resolution": [width, height],
            "processed_resolution": list(FRAME_SIZE) if FRAME_SIZE else [width, height],
            "frames_requested": num_frames,
            "frames_extracted": successful_extractions,
            "frames": extracted_frames
        }
        
        metadata_path = frames_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Extracted {successful_extractions}/{num_frames} frames to {frames_dir}")
        
        return {
            "success": True,
            "video_id": video_id,
            "frames_dir": str(frames_dir),
            "frames_extracted": successful_extractions,
            "frame_paths": [f["path"] for f in extracted_frames],
            "metadata": metadata
        }
        
    except Exception as e:
        print(f"❌ Frame extraction failed for {video_id}: {str(e)}")
        return {
            "success": False,
            "video_id": video_id,
            "error": str(e)
        }

def load_frames(video_id: str) -> Optional[List[np.ndarray]]:
    """
    Load extracted frames for a video as numpy arrays
    
    Args:
        video_id: Unique video identifier
    
    Returns:
        List of frame arrays or None if not found
    """
    frames_dir = DATA_DIR / "frames" / video_id
    metadata_path = frames_dir / "metadata.json"
    
    if not metadata_path.exists():
        print(f"⚠️ No frame metadata found for video {video_id}")
        return None
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        frames = []
        for frame_info in metadata["frames"]:
            frame_path = frame_info["path"]
            if Path(frame_path).exists():
                frame = cv2.imread(frame_path)
                if frame is not None:
                    frames.append(frame)
                else:
                    print(f"⚠️ Could not load frame: {frame_path}")
            else:
                print(f"⚠️ Frame file not found: {frame_path}")
        
        print(f"📸 Loaded {len(frames)} frames for video {video_id}")
        return frames
        
    except Exception as e:
        print(f"❌ Failed to load frames for {video_id}: {str(e)}")
        return None

def get_frame_paths(video_id: str) -> Optional[List[str]]:
    """
    Get list of frame file paths for a video
    
    Args:
        video_id: Unique video identifier
    
    Returns:
        List of frame file paths or None if not found
    """
    frames_dir = DATA_DIR / "frames" / video_id
    metadata_path = frames_dir / "metadata.json"
    
    if not metadata_path.exists():
        return None
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return [frame_info["path"] for frame_info in metadata["frames"]]
        
    except Exception as e:
        print(f"❌ Failed to get frame paths for {video_id}: {str(e)}")
        return None

def extract_frames_batch(video_list: List[Dict]) -> Dict:
    """
    Extract frames from multiple videos
    
    Args:
        video_list: List of dicts with 'video_id' and 'filepath' keys
    
    Returns:
        Summary of extraction results
    """
    print(f"🎬 Starting batch frame extraction for {len(video_list)} videos")
    
    results = {
        "successful": [],
        "failed": [],
        "skipped": []
    }
    
    for i, video_info in enumerate(video_list):
        video_id = video_info["video_id"]
        filepath = video_info["filepath"]
        
        print(f"\n📹 Processing {i+1}/{len(video_list)}: {video_id}")
        
        # Check if frames already extracted
        frames_dir = DATA_DIR / "frames" / video_id
        if frames_dir.exists() and (frames_dir / "metadata.json").exists():
            print(f"⚠️ Frames already extracted for {video_id}, skipping")
            results["skipped"].append({
                "video_id": video_id,
                "frames_dir": str(frames_dir)
            })
            continue
        
        # Extract frames
        result = extract_frames(filepath, video_id)
        
        if result["success"]:
            results["successful"].append(result)
        else:
            results["failed"].append(result)
    
    print(f"\n📊 FRAME EXTRACTION SUMMARY:")
    print(f"✅ Successful: {len(results['successful'])}")
    print(f"⚠️ Skipped: {len(results['skipped'])}")
    print(f"❌ Failed: {len(results['failed'])}")
    
    return results

def main():
    """Test frame extraction with a sample video"""
    # Look for any video in the raw_videos directory
    raw_videos_dir = DATA_DIR / "raw_videos"
    video_files = list(raw_videos_dir.glob("*.*"))
    
    if not video_files:
        print("❌ No videos found in raw_videos directory")
        print("Run download_videos.py first to download some test videos")
        return
    
    # Test with first video found
    test_video = video_files[0]
    video_id = test_video.stem
    
    print(f"🧪 Testing frame extraction with: {test_video.name}")
    result = extract_frames(str(test_video), video_id)
    
    if result["success"]:
        print(f"✅ Test successful! Extracted {result['frames_extracted']} frames")
        
        # Test loading frames
        frames = load_frames(video_id)
        if frames:
            print(f"✅ Successfully loaded {len(frames)} frames as numpy arrays")
        else:
            print("❌ Failed to load frames")
    else:
        print(f"❌ Test failed: {result['error']}")

if __name__ == "__main__":
    main()
