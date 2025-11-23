"""
Frequency Domain Features
2D FFT analysis for AI detection
"""

import cv2
import numpy as np
from typing import Dict, List

def compute_frequency_features(frames: List[np.ndarray]) -> Dict:
    """Compute frequency domain features from video frames"""
    try:
        # Sample a few frames for frequency analysis
        sample_frames = frames[::max(1, len(frames)//4)][:4]
        
        freq_features = []
        
        for frame in sample_frames:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            gray = cv2.resize(gray, (128, 128))
            
            # Compute 2D FFT
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)
            
            # Divide into frequency bands
            h, w = magnitude_spectrum.shape
            center_h, center_w = h // 2, w // 2
            
            # Low frequency (center 10%)
            low_mask = np.zeros((h, w))
            low_size = min(h, w) // 10
            low_mask[center_h-low_size:center_h+low_size, center_w-low_size:center_w+low_size] = 1
            low_power = np.sum(magnitude_spectrum * low_mask)
            
            # High frequency (outer regions)
            high_mask = 1 - low_mask
            high_power = np.sum(magnitude_spectrum * high_mask)
            
            # Mid frequency (between low and high)
            mid_mask = np.zeros((h, w))
            mid_size = min(h, w) // 4
            mid_mask[center_h-mid_size:center_h+mid_size, center_w-mid_size:center_w+mid_size] = 1
            mid_mask = mid_mask - low_mask
            mid_power = np.sum(magnitude_spectrum * mid_mask)
            
            freq_features.append({
                'low_power': low_power,
                'mid_power': mid_power,
                'high_power': high_power,
                'high_low_ratio': high_power / (low_power + 1e-6),
                'spectrum_slope': -1.0  # Placeholder for log-log slope
            })
        
        # Aggregate across frames
        return {
            "freq_low_power_mean": float(np.mean([f['low_power'] for f in freq_features])),
            "freq_mid_power_mean": float(np.mean([f['mid_power'] for f in freq_features])),
            "freq_high_power_mean": float(np.mean([f['high_power'] for f in freq_features])),
            "freq_high_low_ratio_mean": float(np.mean([f['high_low_ratio'] for f in freq_features])),
            "freq_spectrum_slope_mean": float(np.mean([f['spectrum_slope'] for f in freq_features]))
        }
        
    except Exception as e:
        print(f"⚠️ Frequency computation failed: {e}")
        return {
            "freq_low_power_mean": 0.0,
            "freq_mid_power_mean": 0.0,
            "freq_high_power_mean": 0.0,
            "freq_high_low_ratio_mean": 0.0,
            "freq_spectrum_slope_mean": 0.0
        }
