"""
Basic Metadata Features
Video properties for AI detection
"""

import cv2
from typing import Dict

def compute_metadata_features(video_path: str) -> Dict:
    """Compute basic metadata features from video file"""
    try:
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate duration
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            "duration_seconds": float(duration),
            "fps": float(fps),
            "resolution_width": int(width),
            "resolution_height": int(height)
        }
        
    except Exception as e:
        print(f"⚠️ Metadata extraction failed: {e}")
        return {
            "duration_seconds": 0.0,
            "fps": 0.0,
            "resolution_width": 0,
            "resolution_height": 0
        }
