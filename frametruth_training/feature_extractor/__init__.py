"""
Feature Extraction Package for FrameTruth Training Pipeline
Master API for computing all forensic features from videos
"""

import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import cv2
import json

from config import DATA_DIR
from frame_extractor import load_frames, get_frame_paths

# Import feature modules
from .prnu_features import compute_prnu_features
from .trajectory_features import compute_trajectory_features
from .optical_flow_features import compute_optical_flow_features
from .ocr_features import compute_ocr_features
from .frequency_features import compute_frequency_features
from .noise_features import compute_noise_features
from .codec_features import compute_codec_features
from .metadata_features import compute_metadata_features

def compute_all_features(video_id: str, video_path: str) -> Dict:
    """
    Master function to compute all forensic features for a video
    
    Args:
        video_id: Unique video identifier
        video_path: Path to video file
    
    Returns:
        Dictionary with all computed features
    """
    print(f"🔬 Computing all features for video {video_id}")
    
    # Initialize feature dictionary
    features = {
        "video_id": video_id,
        "video_path": video_path
    }
    
    # Load frames for analysis
    frames = load_frames(video_id)
    frame_paths = get_frame_paths(video_id)
    
    if not frames or not frame_paths:
        print(f"❌ No frames found for video {video_id}")
        return {"error": "No frames available", "video_id": video_id}
    
    print(f"📸 Loaded {len(frames)} frames for analysis")
    
    # 1. PRNU Sensor Fingerprint Features
    print("🔬 Computing PRNU sensor fingerprint features...")
    try:
        prnu_features = compute_prnu_features(frames)
        features.update(prnu_features)
        print(f"✅ PRNU features: {list(prnu_features.keys())}")
    except Exception as e:
        print(f"⚠️ PRNU feature extraction failed: {e}")
        features.update({
            "prnu_mean_corr": 0.0,
            "prnu_std_corr": 0.0,
            "prnu_positive_ratio": 0.0,
            "prnu_consistency_score": 0.0
        })
    
    # 2. Trajectory/Curvature Features
    print("📐 Computing trajectory curvature features...")
    try:
        trajectory_features = compute_trajectory_features(frames)
        features.update(trajectory_features)
        print(f"✅ Trajectory features: {list(trajectory_features.keys())}")
    except Exception as e:
        print(f"⚠️ Trajectory feature extraction failed: {e}")
        features.update({
            "trajectory_curvature_mean": 0.0,
            "trajectory_curvature_std": 0.0,
            "trajectory_max_curvature": 0.0,
            "trajectory_mean_distance": 0.0
        })
    
    # 3. Optical Flow Features
    print("🌊 Computing optical flow features...")
    try:
        flow_features = compute_optical_flow_features(frames)
        features.update(flow_features)
        print(f"✅ Optical flow features: {list(flow_features.keys())}")
    except Exception as e:
        print(f"⚠️ Optical flow feature extraction failed: {e}")
        features.update({
            "flow_jitter_index": 0.0,
            "flow_bg_fg_ratio": 0.0,
            "flow_patch_variance": 0.0,
            "flow_smoothness_score": 0.0,
            "flow_global_mean": 0.0,
            "flow_global_std": 0.0
        })
    
    # 4. OCR Text Stability Features
    print("📝 Computing OCR text stability features...")
    try:
        ocr_features = compute_ocr_features(frames)
        features.update(ocr_features)
        print(f"✅ OCR features: {list(ocr_features.keys())}")
    except Exception as e:
        print(f"⚠️ OCR feature extraction failed: {e}")
        features.update({
            "ocr_char_error_rate": 0.0,
            "ocr_frame_stability": 1.0,
            "ocr_mutation_rate": 0.0,
            "ocr_unique_string_count": 0,
            "ocr_total_detections": 0
        })
    
    # 5. Frequency Domain Features (NEW)
    print("📊 Computing frequency domain features...")
    try:
        freq_features = compute_frequency_features(frames)
        features.update(freq_features)
        print(f"✅ Frequency features: {list(freq_features.keys())}")
    except Exception as e:
        print(f"⚠️ Frequency feature extraction failed: {e}")
        features.update({
            "freq_low_power_mean": 0.0,
            "freq_mid_power_mean": 0.0,
            "freq_high_power_mean": 0.0,
            "freq_high_low_ratio_mean": 0.0,
            "freq_spectrum_slope_mean": 0.0
        })
    
    # 6. Noise/Grain Statistics Features (NEW)
    print("🔍 Computing noise/grain statistics features...")
    try:
        noise_features = compute_noise_features(frames)
        features.update(noise_features)
        print(f"✅ Noise features: {list(noise_features.keys())}")
    except Exception as e:
        print(f"⚠️ Noise feature extraction failed: {e}")
        features.update({
            "noise_variance_r_mean": 0.0,
            "noise_variance_g_mean": 0.0,
            "noise_variance_b_mean": 0.0,
            "cross_channel_corr_rg_mean": 0.0,
            "spatial_autocorr_mean": 0.0,
            "temporal_noise_consistency": 0.0
        })
    
    # 7. Codec/GOP Features (NEW)
    print("🎥 Computing codec/GOP features...")
    try:
        codec_features = compute_codec_features(video_path)
        features.update(codec_features)
        print(f"✅ Codec features: {list(codec_features.keys())}")
    except Exception as e:
        print(f"⚠️ Codec feature extraction failed: {e}")
        features.update({
            "avg_bitrate": 0.0,
            "i_frame_ratio": 0.0,
            "p_frame_ratio": 0.0,
            "b_frame_ratio": 0.0,
            "gop_length_mean": 0.0,
            "gop_length_std": 0.0,
            "double_compression_score": 0.0
        })
    
    # 8. Basic Metadata Features
    print("📋 Computing metadata features...")
    try:
        metadata_features = compute_metadata_features(video_path)
        features.update(metadata_features)
        print(f"✅ Metadata features: {list(metadata_features.keys())}")
    except Exception as e:
        print(f"⚠️ Metadata feature extraction failed: {e}")
        features.update({
            "duration_seconds": 0.0,
            "fps": 0.0,
            "resolution_width": 0,
            "resolution_height": 0
        })
    
    print(f"🎯 Feature extraction complete for {video_id}")
    print(f"📊 Total features extracted: {len([k for k in features.keys() if k not in ['video_id', 'video_path']])}")
    
    return features

def extract_features_batch(video_list: List[Dict]) -> List[Dict]:
    """
    Extract features from multiple videos
    
    Args:
        video_list: List of dicts with video info
    
    Returns:
        List of feature dictionaries
    """
    print(f"🔬 Starting batch feature extraction for {len(video_list)} videos")
    
    all_features = []
    
    for i, video_info in enumerate(video_list):
        video_id = video_info["video_id"]
        video_path = video_info["filepath"]
        
        print(f"\n🎯 Processing {i+1}/{len(video_list)}: {video_id}")
        
        features = compute_all_features(video_id, video_path)
        
        if "error" not in features:
            # Add label if available
            if "label" in video_info:
                features["label"] = video_info["label"]
            
            all_features.append(features)
        else:
            print(f"❌ Skipping {video_id} due to feature extraction failure")
    
    print(f"\n📊 FEATURE EXTRACTION SUMMARY:")
    print(f"✅ Successful: {len(all_features)}")
    print(f"❌ Failed: {len(video_list) - len(all_features)}")
    
    return all_features

def save_features_to_csv(features_list: List[Dict], output_path: str):
    """
    Save extracted features to CSV file
    
    Args:
        features_list: List of feature dictionaries
        output_path: Path to save CSV file
    """
    import pandas as pd
    
    if not features_list:
        print("❌ No features to save")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(features_list)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"💾 Saved {len(features_list)} feature vectors to {output_path}")
    print(f"📊 Feature columns: {len(df.columns)}")
    print(f"📋 Sample features: {list(df.columns)[:10]}...")

def main():
    """Test feature extraction with available videos"""
    # Look for extracted frames
    frames_dir = DATA_DIR / "frames"
    video_dirs = [d for d in frames_dir.iterdir() if d.is_dir()]
    
    if not video_dirs:
        print("❌ No extracted frames found")
        print("Run frame_extractor.py first to extract frames from videos")
        return
    
    # Test with first video found
    test_video_id = video_dirs[0].name
    
    # Find corresponding video file
    raw_videos_dir = DATA_DIR / "raw_videos"
    video_files = list(raw_videos_dir.glob(f"{test_video_id}.*"))
    
    if not video_files:
        print(f"❌ No video file found for {test_video_id}")
        return
    
    test_video_path = str(video_files[0])
    
    print(f"🧪 Testing feature extraction with: {test_video_id}")
    features = compute_all_features(test_video_id, test_video_path)
    
    if "error" not in features:
        print(f"✅ Test successful! Extracted {len(features)} features")
        print(f"📊 Sample features:")
        for key, value in list(features.items())[:10]:
            print(f"  {key}: {value}")
    else:
        print(f"❌ Test failed: {features['error']}")

if __name__ == "__main__":
    main()
