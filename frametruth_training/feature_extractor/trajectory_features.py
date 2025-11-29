"""
Enhanced ReStraV Trajectory Analysis
Full implementation of ReStraV methodology with DINOv2 embeddings
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import Dict, List, Optional, Tuple

def compute_trajectory_features(frames: List[np.ndarray], dinov2_model=None, dinov2_transform=None) -> Dict:
    """
    Compute enhanced trajectory curvature features using ReStraV methodology
    
    Args:
        frames: List of video frames as numpy arrays
        dinov2_model: Pre-loaded DINOv2 model (optional, will use lightweight if None)
        dinov2_transform: DINOv2 preprocessing transform (optional)
    
    Returns:
        Dictionary with trajectory metrics and ReStraV score
    """
    if len(frames) < 3:
        return _get_default_trajectory_result()
    
    try:
        # Use DINOv2 embeddings if available, otherwise fall back to lightweight features
        if dinov2_model is not None and dinov2_transform is not None:
            features = _extract_dinov2_features(frames, dinov2_model, dinov2_transform)
            method = "dinov2_restrav"
        else:
            features = _extract_lightweight_features(frames)
            method = "lightweight_restrav"
        
        if len(features) < 3:
            return _get_default_trajectory_result()
        
        # Calculate ReStraV trajectory metrics
        trajectory_metrics = _calculate_restrav_metrics(features)
        trajectory_metrics['method'] = method
        trajectory_metrics['num_frames'] = len(frames)
        
        # Calculate ReStraV confidence score (0-100)
        restrav_score = _calculate_restrav_score(trajectory_metrics)
        trajectory_metrics['restrav_confidence'] = restrav_score
        
        return trajectory_metrics
        
    except Exception as e:
        print(f"⚠️ Enhanced trajectory computation failed: {e}")
        return _get_default_trajectory_result()

def _extract_dinov2_features(frames: List[np.ndarray], model, transform) -> List[np.ndarray]:
    """Extract DINOv2 embeddings from frames"""
    features = []
    
    with torch.no_grad():
        for frame in frames:
            try:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                
                # Apply DINOv2 preprocessing
                tensor = transform(pil_image).unsqueeze(0)
                
                # Extract features
                embedding = model(tensor)
                
                # Convert to numpy and flatten
                feature_vec = embedding.squeeze().cpu().numpy()
                features.append(feature_vec)
                
            except Exception as e:
                print(f"⚠️ DINOv2 feature extraction failed for frame: {e}")
                continue
    
    return features

def _extract_lightweight_features(frames: List[np.ndarray]) -> List[np.ndarray]:
    """Extract lightweight visual features as fallback"""
    features = []
    
    for frame in frames:
        try:
            # Resize for processing
            frame_small = cv2.resize(frame, (224, 224))  # Larger for better features
            
            # Multi-scale color histograms
            hist_features = []
            for scale in [(64, 64), (128, 128)]:
                scaled = cv2.resize(frame_small, scale)
                
                # HSV color histogram (more robust than RGB)
                hsv = cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)
                hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
                hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
                hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
                
                scale_hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
                scale_hist = scale_hist / (scale_hist.sum() + 1e-6)
                hist_features.append(scale_hist)
            
            # Texture features
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            
            # Local Binary Pattern approximation
            lbp_features = []
            for radius in [1, 2]:
                kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) * radius
                lbp = cv2.filter2D(gray, -1, kernel)
                lbp_hist = cv2.calcHist([lbp.astype(np.uint8)], [0], None, [16], [0, 256])
                lbp_hist = lbp_hist.flatten() / (lbp_hist.sum() + 1e-6)
                lbp_features.append(lbp_hist)
            
            # Edge orientation histogram
            edges = cv2.Canny(gray, 50, 150)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            orientation = np.arctan2(sobel_y, sobel_x)
            edge_hist = np.histogram(orientation[edges > 0], bins=16, range=(-np.pi, np.pi))[0]
            edge_hist = edge_hist / (edge_hist.sum() + 1e-6)
            
            # Combine all features
            feature_vec = np.concatenate([
                *hist_features,  # Multi-scale color histograms
                *lbp_features,   # Texture features
                edge_hist        # Edge orientation
            ])
            
            features.append(feature_vec)
            
        except Exception as e:
            print(f"⚠️ Lightweight feature extraction failed for frame: {e}")
            continue
    
    return features

def _calculate_restrav_metrics(features: List[np.ndarray]) -> Dict:
    """Calculate ReStraV trajectory metrics from feature vectors"""
    features_array = np.array(features)
    
    # Calculate displacement vectors (feature changes between frames)
    displacements = np.diff(features_array, axis=0)
    
    # Calculate stepwise distances (magnitude of change)
    distances = np.linalg.norm(displacements, axis=1)
    
    # Calculate curvature (angle between consecutive displacement vectors)
    curvatures = []
    for i in range(len(displacements) - 1):
        vec1 = displacements[i]
        vec2 = displacements[i + 1]
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 > 1e-6 and norm2 > 1e-6:
            # Calculate cosine similarity
            cos_sim = np.dot(vec1, vec2) / (norm1 * norm2)
            cos_sim = np.clip(cos_sim, -1.0, 1.0)
            # Convert to angle in degrees
            angle = np.arccos(cos_sim) * 180 / np.pi
            curvatures.append(angle)
    
    if len(curvatures) == 0:
        return _get_default_trajectory_result()
    
    # Enhanced ReStraV metrics
    curvature_mean = float(np.mean(curvatures))
    curvature_std = float(np.std(curvatures))
    curvature_max = float(np.max(curvatures))
    distance_mean = float(np.mean(distances))
    distance_std = float(np.std(distances))
    
    # Additional ReStraV metrics
    curvature_variance = float(np.var(curvatures))
    distance_variance = float(np.var(distances))
    
    # Trajectory smoothness (inverse of curvature variance)
    smoothness = 1.0 / (curvature_variance + 1e-6)
    
    # Trajectory consistency (low distance variance indicates consistent motion)
    consistency = 1.0 / (distance_variance + 1e-6)
    
    return {
        "trajectory_curvature_mean": curvature_mean,
        "trajectory_curvature_std": curvature_std,
        "trajectory_curvature_max": curvature_max,
        "trajectory_curvature_variance": curvature_variance,
        "trajectory_distance_mean": distance_mean,
        "trajectory_distance_std": distance_std,
        "trajectory_distance_variance": distance_variance,
        "trajectory_smoothness": float(smoothness),
        "trajectory_consistency": float(consistency)
    }

def _calculate_restrav_score(metrics: Dict) -> float:
    """
    Calculate ReStraV confidence score (0-100) based on trajectory metrics
    
    Real videos: More consistent, smoother trajectories
    AI videos: Irregular, jumpy trajectories with high curvature variance
    """
    try:
        # Extract key metrics
        curvature_mean = metrics.get("trajectory_curvature_mean", 0)
        curvature_variance = metrics.get("trajectory_curvature_variance", 0)
        distance_variance = metrics.get("trajectory_distance_variance", 0)
        smoothness = metrics.get("trajectory_smoothness", 0)
        consistency = metrics.get("trajectory_consistency", 0)
        
        # ReStraV scoring logic (based on research findings)
        # AI videos typically have:
        # - Higher curvature variance (irregular motion)
        # - Higher distance variance (inconsistent feature changes)
        # - Lower smoothness and consistency
        
        ai_indicators = 0.0
        
        # High curvature variance indicates AI (irregular motion patterns)
        if curvature_variance > 800:  # Threshold based on empirical analysis
            ai_indicators += 25
        elif curvature_variance > 400:
            ai_indicators += 15
        elif curvature_variance > 200:
            ai_indicators += 5
        
        # High distance variance indicates AI (inconsistent feature evolution)
        if distance_variance > 0.1:
            ai_indicators += 20
        elif distance_variance > 0.05:
            ai_indicators += 10
        
        # Low smoothness indicates AI
        if smoothness < 0.001:
            ai_indicators += 20
        elif smoothness < 0.01:
            ai_indicators += 10
        
        # Low consistency indicates AI
        if consistency < 10:
            ai_indicators += 15
        elif consistency < 50:
            ai_indicators += 5
        
        # Very high curvature mean can indicate AI (unnatural motion)
        if curvature_mean > 120:
            ai_indicators += 10
        elif curvature_mean > 90:
            ai_indicators += 5
        
        # Cap at 100
        ai_score = min(100, ai_indicators)
        
        return float(ai_score)
        
    except Exception as e:
        print(f"⚠️ ReStraV score calculation failed: {e}")
        return 0.0

def _get_default_trajectory_result() -> Dict:
    """Return default trajectory result for error cases"""
    return {
        "trajectory_curvature_mean": 0.0,
        "trajectory_curvature_std": 0.0,
        "trajectory_curvature_max": 0.0,
        "trajectory_curvature_variance": 0.0,
        "trajectory_distance_mean": 0.0,
        "trajectory_distance_std": 0.0,
        "trajectory_distance_variance": 0.0,
        "trajectory_smoothness": 0.0,
        "trajectory_consistency": 0.0,
        "restrav_confidence": 0.0,
        "method": "error",
        "num_frames": 0
    }

# Legacy function for backward compatibility
def compute_lightweight_trajectory_features(frames: List[np.ndarray]) -> Dict:
    """Legacy function - use compute_trajectory_features instead"""
    return compute_trajectory_features(frames)
