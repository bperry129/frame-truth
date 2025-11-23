"""
Optical Flow Features
Motion analysis for AI detection
"""

import cv2
import numpy as np
from typing import Dict, List

def compute_optical_flow_features(frames: List[np.ndarray]) -> Dict:
    """Compute optical flow features from video frames"""
    if len(frames) < 2:
        return {
            "flow_jitter_index": 0.0,
            "flow_bg_fg_ratio": 0.0,
            "flow_patch_variance": 0.0,
            "flow_smoothness_score": 0.0,
            "flow_global_mean": 0.0,
            "flow_global_std": 0.0
        }
    
    try:
        # Convert to grayscale and resize
        gray_frames = []
        for frame in frames[:6]:  # Limit for speed
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            gray_small = cv2.resize(gray, (128, 128))
            gray_frames.append(gray_small)
        
        flow_magnitudes = []
        flow_directions = []
        
        for i in range(len(gray_frames) - 1):
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[i], gray_frames[i+1], None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_magnitudes.append(np.mean(magnitude))
            flow_directions.append(np.mean(angle))
        
        if not flow_magnitudes:
            return {
                "flow_jitter_index": 0.0,
                "flow_bg_fg_ratio": 0.0,
                "flow_patch_variance": 0.0,
                "flow_smoothness_score": 0.0,
                "flow_global_mean": 0.0,
                "flow_global_std": 0.0
            }
        
        # Calculate features
        flow_global_mean = float(np.mean(flow_magnitudes))
        flow_global_std = float(np.std(flow_magnitudes))
        
        # Jitter index (direction changes)
        direction_changes = []
        for i in range(len(flow_directions) - 1):
            diff = abs(flow_directions[i+1] - flow_directions[i])
            diff = min(diff, 2*np.pi - diff)
            direction_changes.append(diff)
        
        flow_jitter_index = float(np.mean(direction_changes)) if direction_changes else 0.0
        
        # Simple background/foreground ratio
        flow_bg_fg_ratio = 0.5  # Placeholder
        flow_patch_variance = flow_global_std  # Simplified
        flow_smoothness_score = 1.0 / (flow_global_std + 1e-6)
        
        return {
            "flow_jitter_index": flow_jitter_index,
            "flow_bg_fg_ratio": flow_bg_fg_ratio,
            "flow_patch_variance": flow_patch_variance,
            "flow_smoothness_score": flow_smoothness_score,
            "flow_global_mean": flow_global_mean,
            "flow_global_std": flow_global_std
        }
        
    except Exception as e:
        print(f"⚠️ Optical flow computation failed: {e}")
        return {
            "flow_jitter_index": 0.0,
            "flow_bg_fg_ratio": 0.0,
            "flow_patch_variance": 0.0,
            "flow_smoothness_score": 0.0,
            "flow_global_mean": 0.0,
            "flow_global_std": 0.0
        }
