"""
Codec/GOP Features
Video compression analysis for AI detection
"""

import cv2
import subprocess
import json
from typing import Dict

def compute_codec_features(video_path: str) -> Dict:
    """Compute codec and GOP features from video file"""
    try:
        # Try to use ffprobe to get detailed video info
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            
            # Extract video stream info
            video_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if video_stream:
                # Extract codec features
                codec_name = video_stream.get('codec_name', 'unknown')
                bit_rate = float(video_stream.get('bit_rate', 0))
                
                # GOP analysis (simplified - would need more complex analysis for real GOP detection)
                # For now, use placeholder values
                return {
                    "avg_bitrate": bit_rate,
                    "i_frame_ratio": 0.1,  # Placeholder
                    "p_frame_ratio": 0.7,  # Placeholder
                    "b_frame_ratio": 0.2,  # Placeholder
                    "gop_length_mean": 15.0,  # Placeholder
                    "gop_length_std": 2.0,   # Placeholder
                    "double_compression_score": 0.0  # Placeholder
                }
        
        # Fallback: use OpenCV to get basic info
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Estimate bitrate from file size and duration
        import os
        file_size = os.path.getsize(video_path)
        duration = frame_count / fps if fps > 0 else 1
        estimated_bitrate = (file_size * 8) / duration  # bits per second
        
        return {
            "avg_bitrate": float(estimated_bitrate),
            "i_frame_ratio": 0.1,  # Default estimates
            "p_frame_ratio": 0.7,
            "b_frame_ratio": 0.2,
            "gop_length_mean": 15.0,
            "gop_length_std": 2.0,
            "double_compression_score": 0.0
        }
        
    except Exception as e:
        print(f"⚠️ Codec analysis failed: {e}")
        return {
            "avg_bitrate": 0.0,
            "i_frame_ratio": 0.0,
            "p_frame_ratio": 0.0,
            "b_frame_ratio": 0.0,
            "gop_length_mean": 0.0,
            "gop_length_std": 0.0,
            "double_compression_score": 0.0
        }
