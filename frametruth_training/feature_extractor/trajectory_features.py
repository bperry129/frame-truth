"""
Trajectory/Curvature Features
ReStraV-inspired visual trajectory analysis
"""

import cv2
import numpy as np
from typing import Dict, List

def compute_trajectory_features(frames: List[np.ndarray]) -> Dict:
    """
    Compute trajectory curvature features from video frames
    Based on ReStraV methodology - analyzes visual feature trajectories
    """
    if len(frames) < 3:
        return {
            "trajectory_curvature_mean": 0.0,
            "trajectory_curvature_std": 0.0,
            "trajectory_max_curvature": 0.0,
            "trajectory_mean_distance": 0.0
        }
    
    try:
        # Extract visual features from each frame
        features = []
        for frame in frames:
            # Resize for faster processing
            frame_small = cv2.resize(frame, (128, 128))
            
            # Extract color histogram
            hist_r = cv2.calcHist([frame_small], [0], None, [32], [0, 256])
            hist_g = cv2.calcHist([frame_small], [1], None, [32], [0, 256])
            hist_b = cv2.calcHist([frame_small], [2], None, [32], [0, 256])
            color_hist = np.concatenate([hist_r, hist_g, hist_b]).flatten()
            color_hist = color_hist / (color_hist.sum() + 1e-6)
            
            # Extract edge density
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.mean(edges) / 255.0
            
            # Extract brightness variance
            brightness_var = np.var(gray) / 255.0
            
            # Combine features
            feature_vec = np.concatenate([color_hist, [edge_density, brightness_var]])
            features.append(feature_vec)
        
        # Calculate trajectory curvature
        features_array = np.array(features)
        displacements = np.diff(features_array, axis=0)
        distances = np.linalg.norm(displacements, axis=1)
        
        # Calculate curvature angles
        curvatures = []
        for i in range(len(displacements) - 1):
            vec1 = displacements[i]
            vec2 = displacements[i + 1]
            
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 > 1e-6 and norm2 > 1e-6:
                cos_sim = np.dot(vec1, vec2) / (norm1 * norm2)
                cos_sim = np.clip(cos_sim, -1.0, 1.0)
                angle = np.arccos(cos_sim) * 180 / np.pi
                curvatures.append(angle)
        
        if len(curvatures) == 0:
            return {
                "trajectory_curvature_mean": 0.0,
                "trajectory_curvature_std": 0.0,
                "trajectory_max_curvature": 0.0,
                "trajectory_mean_distance": 0.0
            }
        
        return {
            "trajectory_curvature_mean": float(np.mean(curvatures)),
            "trajectory_curvature_std": float(np.std(curvatures)),
            "trajectory_max_curvature": float(np.max(curvatures)),
            "trajectory_mean_distance": float(np.mean(distances))
        }
        
    except Exception as e:
        print(f"⚠️ Trajectory computation failed: {e}")
        return {
            "trajectory_curvature_mean": 0.0,
            "trajectory_curvature_std": 0.0,
            "trajectory_max_curvature": 0.0,
            "trajectory_mean_distance": 0.0
        }
