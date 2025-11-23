"""
PRNU (Photo-Response Non-Uniformity) Sensor Fingerprint Features
Camera sensor "DNA" analysis - the most reliable AI detection signal
"""

import cv2
import numpy as np
from typing import Dict, List

def compute_prnu_features(frames: List[np.ndarray]) -> Dict:
    """
    Compute PRNU sensor fingerprint features from video frames
    
    Real cameras have consistent sensor noise patterns (fingerprint)
    AI videos lack this physical sensor consistency
    
    Args:
        frames: List of video frames as numpy arrays
    
    Returns:
        Dictionary with PRNU features
    """
    if len(frames) < 4:
        # Need minimum frames for reliable fingerprint
        return {
            "prnu_mean_corr": 0.0,
            "prnu_std_corr": 0.0,
            "prnu_positive_ratio": 0.0,
            "prnu_consistency_score": 0.0
        }
    
    try:
        # Convert frames to grayscale and resize for processing
        gray_frames = []
        for frame in frames:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Resize to manageable size for processing speed
            gray_resized = cv2.resize(gray, (256, 256))
            gray_frames.append(gray_resized.astype(np.float32))
        
        # Step 1: Extract noise residuals from each frame
        residuals = []
        for gray_frame in gray_frames:
            # Apply strong denoising to isolate sensor noise
            denoised = cv2.fastNlMeansDenoising(
                gray_frame.astype(np.uint8), 
                None, 
                h=10,           # Filter strength
                templateWindowSize=7,
                searchWindowSize=21
            )
            denoised = denoised.astype(np.float32)
            
            # Calculate noise residual: original - denoised
            residual = gray_frame - denoised
            
            # Normalize residual to unit variance
            residual_std = np.std(residual)
            if residual_std > 1e-6:
                residual = residual / residual_std
            
            residuals.append(residual)
        
        # Step 2: Estimate global sensor fingerprint
        # Average residuals across frames - sensor noise should reinforce
        global_fingerprint = np.mean(residuals, axis=0)
        
        # Normalize the global fingerprint
        fingerprint_std = np.std(global_fingerprint)
        if fingerprint_std > 1e-6:
            global_fingerprint = global_fingerprint / fingerprint_std
        
        # Step 3: Measure per-frame correlation with global fingerprint
        correlations = []
        for residual in residuals:
            # Flatten for correlation calculation
            residual_flat = residual.flatten()
            fingerprint_flat = global_fingerprint.flatten()
            
            # Calculate normalized cross-correlation
            if len(residual_flat) > 0 and len(fingerprint_flat) > 0:
                correlation_matrix = np.corrcoef(residual_flat, fingerprint_flat)
                if correlation_matrix.shape == (2, 2):
                    correlation = correlation_matrix[0, 1]
                else:
                    correlation = 0.0
            else:
                correlation = 0.0
            
            # Handle NaN case
            if np.isnan(correlation):
                correlation = 0.0
            
            correlations.append(correlation)
        
        # Step 4: Calculate PRNU metrics
        correlations = np.array(correlations)
        
        # Mean correlation with global fingerprint
        prnu_mean_corr = float(np.mean(correlations))
        
        # Standard deviation of correlations (consistency measure)
        prnu_std_corr = float(np.std(correlations))
        
        # Ratio of positive correlations (should be high for real cameras)
        positive_threshold = 0.1
        prnu_positive_ratio = float(np.sum(correlations > positive_threshold) / len(correlations))
        
        # Overall consistency score
        # Real cameras: high mean, low std, high positive ratio
        # AI videos: near-zero mean, higher std, low positive ratio
        if prnu_std_corr > 0:
            consistency_score = (prnu_mean_corr * prnu_positive_ratio) / (prnu_std_corr + 0.1)
        else:
            consistency_score = prnu_mean_corr * prnu_positive_ratio
        
        prnu_consistency_score = float(max(0, min(1, consistency_score)))
        
        return {
            "prnu_mean_corr": prnu_mean_corr,
            "prnu_std_corr": prnu_std_corr,
            "prnu_positive_ratio": prnu_positive_ratio,
            "prnu_consistency_score": prnu_consistency_score
        }
        
    except Exception as e:
        print(f"⚠️ PRNU computation failed: {e}")
        return {
            "prnu_mean_corr": 0.0,
            "prnu_std_corr": 0.0,
            "prnu_positive_ratio": 0.0,
            "prnu_consistency_score": 0.0
        }

def main():
    """Test PRNU feature extraction"""
    # Create synthetic test frames
    print("🧪 Testing PRNU feature extraction...")
    
    # Simulate real camera frames (with consistent noise pattern)
    real_frames = []
    base_noise = np.random.normal(0, 0.1, (256, 256)).astype(np.float32)
    
    for i in range(8):
        # Base image with some content
        frame = np.random.uniform(50, 200, (256, 256)).astype(np.float32)
        # Add consistent sensor noise pattern
        frame += base_noise * 0.5 + np.random.normal(0, 0.05, (256, 256))
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        real_frames.append(frame)
    
    real_features = compute_prnu_features(real_frames)
    print(f"📊 Real camera simulation:")
    for key, value in real_features.items():
        print(f"  {key}: {value:.4f}")
    
    # Simulate AI frames (with inconsistent noise)
    ai_frames = []
    for i in range(8):
        # Base image with some content
        frame = np.random.uniform(50, 200, (256, 256)).astype(np.float32)
        # Add random noise (no consistent pattern)
        frame += np.random.normal(0, 0.1, (256, 256))
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        ai_frames.append(frame)
    
    ai_features = compute_prnu_features(ai_frames)
    print(f"\n📊 AI video simulation:")
    for key, value in ai_features.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\n🎯 Expected pattern:")
    print(f"  Real videos should have higher mean_corr and positive_ratio")
    print(f"  AI videos should have lower mean_corr and consistency_score")

if __name__ == "__main__":
    main()
