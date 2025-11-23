"""
Noise/Grain Statistics Features
Flat patch noise analysis for AI detection
"""

import cv2
import numpy as np
from typing import Dict, List

def compute_noise_features(frames: List[np.ndarray]) -> Dict:
    """Compute noise/grain statistics from video frames"""
    try:
        # Sample frames for noise analysis
        sample_frames = frames[::max(1, len(frames)//4)][:4]
        
        noise_stats = []
        
        for frame in sample_frames:
            # Resize for processing
            frame_resized = cv2.resize(frame, (128, 128))
            
            # Split into channels
            if len(frame_resized.shape) == 3:
                b, g, r = cv2.split(frame_resized.astype(np.float32))
            else:
                r = g = b = frame_resized.astype(np.float32)
            
            # Find flat patches (low gradient areas)
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY) if len(frame_resized.shape) == 3 else frame_resized
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Select flat patches (low gradient)
            flat_threshold = np.percentile(gradient_magnitude, 25)
            flat_mask = gradient_magnitude < flat_threshold
            
            if np.sum(flat_mask) > 100:  # Enough flat pixels
                # Noise variance per channel
                noise_var_r = np.var(r[flat_mask])
                noise_var_g = np.var(g[flat_mask])
                noise_var_b = np.var(b[flat_mask])
                
                # Cross-channel correlation
                if len(r[flat_mask]) > 10:
                    corr_rg = np.corrcoef(r[flat_mask], g[flat_mask])[0, 1]
                    if np.isnan(corr_rg):
                        corr_rg = 0.0
                else:
                    corr_rg = 0.0
                
                # Spatial autocorrelation (simplified)
                spatial_autocorr = np.corrcoef(gray[flat_mask][:-1], gray[flat_mask][1:])[0, 1]
                if np.isnan(spatial_autocorr):
                    spatial_autocorr = 0.0
            else:
                noise_var_r = noise_var_g = noise_var_b = 0.0
                corr_rg = spatial_autocorr = 0.0
            
            noise_stats.append({
                'noise_var_r': noise_var_r,
                'noise_var_g': noise_var_g,
                'noise_var_b': noise_var_b,
                'corr_rg': corr_rg,
                'spatial_autocorr': spatial_autocorr
            })
        
        # Temporal consistency
        if len(noise_stats) > 1:
            var_r_values = [s['noise_var_r'] for s in noise_stats]
            temporal_consistency = 1.0 / (np.std(var_r_values) + 1e-6)
        else:
            temporal_consistency = 1.0
        
        # Aggregate features
        return {
            "noise_variance_r_mean": float(np.mean([s['noise_var_r'] for s in noise_stats])),
            "noise_variance_g_mean": float(np.mean([s['noise_var_g'] for s in noise_stats])),
            "noise_variance_b_mean": float(np.mean([s['noise_var_b'] for s in noise_stats])),
            "cross_channel_corr_rg_mean": float(np.mean([s['corr_rg'] for s in noise_stats])),
            "spatial_autocorr_mean": float(np.mean([s['spatial_autocorr'] for s in noise_stats])),
            "temporal_noise_consistency": float(temporal_consistency)
        }
        
    except Exception as e:
        print(f"⚠️ Noise computation failed: {e}")
        return {
            "noise_variance_r_mean": 0.0,
            "noise_variance_g_mean": 0.0,
            "noise_variance_b_mean": 0.0,
            "cross_channel_corr_rg_mean": 0.0,
            "spatial_autocorr_mean": 0.0,
            "temporal_noise_consistency": 0.0
        }
