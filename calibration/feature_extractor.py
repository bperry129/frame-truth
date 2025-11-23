#!/usr/bin/env python3
"""
FrameTruth Feature Extraction Pipeline
Extracts all structural features for calibration dataset
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import easyocr
from difflib import SequenceMatcher

@dataclass
class VideoFeatures:
    """All extracted features for a single video"""
    video_id: str
    label: str  # 'real' or 'ai'
    source: str
    
    # Trajectory features (ReStraV-inspired)
    trajectory_curvature_mean: float
    trajectory_curvature_std: float
    trajectory_max_curvature: float
    trajectory_mean_distance: float
    
    # Optical flow features
    flow_jitter_index: float
    flow_bg_fg_ratio: float
    flow_patch_variance: float
    flow_smoothness_score: float
    flow_global_mean: float
    flow_global_std: float
    
    # OCR text features
    ocr_has_text: bool
    ocr_char_error_rate: float
    ocr_frame_stability: float
    ocr_mutation_rate: float
    ocr_unique_string_count: int
    ocr_total_detections: int
    
    # PRNU sensor features
    prnu_mean_correlation: float
    prnu_std_correlation: float
    prnu_positive_ratio: float
    prnu_consistency_score: float
    
    # Video properties
    duration: float
    fps: float
    resolution_width: int
    resolution_height: int
    
    # Processing metadata
    frames_analyzed: int
    extraction_success: bool
    extraction_notes: str

class FeatureExtractor:
    """Extracts all structural features from videos"""
    
    def __init__(self):
        self.ocr_reader = None  # Lazy loaded
    
    def get_ocr_reader(self):
        """Lazy load EasyOCR reader"""
        if self.ocr_reader is None:
            print("🔄 Loading EasyOCR model...")
            self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            print("✅ EasyOCR loaded")
        return self.ocr_reader
    
    def extract_trajectory_features(self, video_path: str, num_samples: int = 12) -> Dict:
        """Extract trajectory curvature features (ReStraV-inspired)"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0:
                cap.release()
                return None
            
            step = max(1, total_frames // num_samples)
            features = []
            count = 0
            extracted = 0
            
            while cap.isOpened() and extracted < num_samples:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if count % step == 0:
                    # Resize for faster processing
                    frame_small = cv2.resize(frame, (128, 128))
                    
                    # Extract visual features
                    hist_r = cv2.calcHist([frame_small], [0], None, [32], [0, 256])
                    hist_g = cv2.calcHist([frame_small], [1], None, [32], [0, 256])
                    hist_b = cv2.calcHist([frame_small], [2], None, [32], [0, 256])
                    color_hist = np.concatenate([hist_r, hist_g, hist_b]).flatten()
                    color_hist = color_hist / (color_hist.sum() + 1e-6)
                    
                    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    edge_density = np.mean(edges) / 255.0
                    brightness_var = np.var(gray) / 255.0
                    
                    feature_vec = np.concatenate([color_hist, [edge_density, brightness_var]])
                    features.append(feature_vec)
                    extracted += 1
                        
                count += 1
            
            cap.release()
            
            if len(features) < 3:
                return None
            
            # Calculate trajectory curvature
            features_array = np.array(features)
            displacements = np.diff(features_array, axis=0)
            distances = np.linalg.norm(displacements, axis=1)
            
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
                return None
            
            return {
                'curvature_mean': float(np.mean(curvatures)),
                'curvature_std': float(np.std(curvatures)),
                'max_curvature': float(np.max(curvatures)),
                'mean_distance': float(np.mean(distances))
            }
            
        except Exception as e:
            print(f"⚠️ Trajectory extraction failed: {e}")
            return None
    
    def extract_optical_flow_features(self, video_path: str, num_samples: int = 6) -> Dict:
        """Extract optical flow features"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0:
                cap.release()
                return None
            
            step = max(1, total_frames // num_samples)
            frames = []
            count = 0
            extracted = 0
            
            while cap.isOpened() and extracted < num_samples:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if count % step == 0:
                    frame_resized = cv2.resize(frame, (128, 128))
                    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                    frames.append(gray)
                    extracted += 1
                        
                count += 1
            
            cap.release()
            
            if len(frames) < 2:
                return None
            
            # Calculate optical flow features
            flow_magnitudes = []
            flow_directions = []
            patch_variances = []
            
            for i in range(len(frames) - 1):
                flow_dense = cv2.calcOpticalFlowFarneback(
                    frames[i], frames[i+1], None, 
                    pyr_scale=0.5, levels=3, winsize=15, 
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                
                magnitude, angle = cv2.cartToPolar(flow_dense[..., 0], flow_dense[..., 1])
                
                flow_magnitudes.append(np.mean(magnitude))
                flow_directions.append(np.mean(angle))
                
                # Local patch variance
                patch_size = 32
                h, w = magnitude.shape
                patch_vars = []
                
                for y in range(0, h - patch_size, patch_size):
                    for x in range(0, w - patch_size, patch_size):
                        patch = magnitude[y:y+patch_size, x:x+patch_size]
                        patch_vars.append(np.var(patch))
                
                patch_variances.append(np.mean(patch_vars))
            
            if len(flow_magnitudes) == 0:
                return None
            
            # Calculate features
            flow_global_mean = float(np.mean(flow_magnitudes))
            flow_global_std = float(np.std(flow_magnitudes))
            flow_patch_variance_mean = float(np.mean(patch_variances))
            
            # Temporal jitter
            direction_changes = []
            for i in range(len(flow_directions) - 1):
                diff = abs(flow_directions[i+1] - flow_directions[i])
                diff = min(diff, 2*np.pi - diff)
                direction_changes.append(diff)
            
            jitter_index = float(np.mean(direction_changes)) if direction_changes else 0.0
            
            # Background vs foreground flow ratio
            all_magnitudes = np.concatenate([cv2.calcOpticalFlowFarneback(
                frames[i], frames[i+1], None, 
                pyr_scale=0.5, levels=3, winsize=15, 
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )[..., 0].flatten() for i in range(min(3, len(frames)-1))])
            
            magnitude_threshold = np.percentile(all_magnitudes, 70)
            background_flow = np.mean(all_magnitudes[all_magnitudes <= magnitude_threshold])
            foreground_flow = np.mean(all_magnitudes[all_magnitudes > magnitude_threshold])
            bg_fg_ratio = float(background_flow / (foreground_flow + 1e-6))
            
            # Smoothness score (inverse of std)
            smoothness_score = 1.0 / (flow_global_std + 1e-6)
            
            return {
                'jitter_index': jitter_index,
                'bg_fg_ratio': bg_fg_ratio,
                'patch_variance': flow_patch_variance_mean,
                'smoothness_score': smoothness_score,
                'global_mean': flow_global_mean,
                'global_std': flow_global_std
            }
            
        except Exception as e:
            print(f"⚠️ Optical flow extraction failed: {e}")
            return None
    
    def extract_ocr_features(self, video_path: str, num_samples: int = 4) -> Dict:
        """Extract OCR text stability features"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0:
                cap.release()
                return None
            
            step = max(1, total_frames // num_samples)
            frames = []
            count = 0
            extracted = 0
            
            while cap.isOpened() and extracted < num_samples:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if count % step == 0:
                    frame_resized = cv2.resize(frame, (256, 256))
                    frames.append(frame_resized)
                    extracted += 1
                        
                count += 1
            
            cap.release()
            
            if len(frames) < 1:
                return None
            
            reader = self.get_ocr_reader()
            
            # Extract text from each frame
            frame_texts = []
            all_detected_strings = []
            
            for i, frame in enumerate(frames):
                try:
                    results = reader.readtext(frame, detail=1)
                    
                    frame_text_data = []
                    for (bbox, text, confidence) in results:
                        if confidence > 0.4:
                            cleaned_text = text.strip()
                            if len(cleaned_text) > 1:
                                frame_text_data.append({
                                    'text': cleaned_text,
                                    'confidence': confidence,
                                    'bbox': bbox,
                                    'frame': i
                                })
                                all_detected_strings.append(cleaned_text)
                    
                    frame_texts.append(frame_text_data)
                    
                except Exception as e:
                    frame_texts.append([])
            
            if not all_detected_strings:
                return {
                    'has_text': False,
                    'char_error_rate': 0.0,
                    'frame_stability': 1.0,
                    'mutation_rate': 0.0,
                    'unique_string_count': 0,
                    'total_detections': 0
                }
            
            # Analyze character anomalies
            char_error_count = 0
            total_chars = 0
            
            for text in all_detected_strings:
                for char in text:
                    total_chars += 1
                    if not char.isprintable() or ord(char) > 127:
                        char_error_count += 1
                    elif char in ['', '□', '▢', '◯', '○']:
                        char_error_count += 1
            
            char_error_rate = float(char_error_count / max(total_chars, 1))
            
            # Frame stability analysis
            unique_strings = list(set(all_detected_strings))
            unique_string_count = len(unique_strings)
            
            string_stability = {}
            for unique_str in unique_strings:
                appearances = []
                for frame_idx, frame_data in enumerate(frame_texts):
                    found_in_frame = False
                    for detection in frame_data:
                        similarity = SequenceMatcher(None, unique_str.lower(), detection['text'].lower()).ratio()
                        if similarity > 0.8:
                            found_in_frame = True
                            break
                    appearances.append(found_in_frame)
                
                if sum(appearances) > 1:
                    consecutive_runs = []
                    current_run = 0
                    for appeared in appearances:
                        if appeared:
                            current_run += 1
                        else:
                            if current_run > 0:
                                consecutive_runs.append(current_run)
                            current_run = 0
                    if current_run > 0:
                        consecutive_runs.append(current_run)
                    
                    max_run = max(consecutive_runs) if consecutive_runs else 0
                    stability_score = max_run / len(appearances)
                    string_stability[unique_str] = stability_score
                else:
                    string_stability[unique_str] = 0.5
            
            frame_stability_score = float(np.mean(list(string_stability.values()))) if string_stability else 1.0
            
            # Text mutation rate
            mutations = 0
            comparisons = 0
            
            for i in range(len(frame_texts) - 1):
                current_texts = set([d['text'].lower() for d in frame_texts[i]])
                next_texts = set([d['text'].lower() for d in frame_texts[i + 1]])
                
                if current_texts or next_texts:
                    comparisons += 1
                    intersection = len(current_texts.intersection(next_texts))
                    union = len(current_texts.union(next_texts))
                    similarity = intersection / max(union, 1)
                    
                    if similarity < 0.7:
                        mutations += 1
            
            mutation_rate = float(mutations / max(comparisons, 1))
            
            return {
                'has_text': True,
                'char_error_rate': char_error_rate,
                'frame_stability': frame_stability_score,
                'mutation_rate': mutation_rate,
                'unique_string_count': unique_string_count,
                'total_detections': len(all_detected_strings)
            }
            
        except Exception as e:
            print(f"⚠️ OCR extraction failed: {e}")
            return None
    
    def extract_prnu_features(self, video_path: str, num_samples: int = 16) -> Dict:
        """Extract PRNU sensor fingerprint features"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0:
                cap.release()
                return None
            
            step = max(1, total_frames // num_samples)
            frames = []
            count = 0
            extracted = 0
            
            while cap.isOpened() and extracted < num_samples:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if count % step == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_resized = cv2.resize(gray, (256, 256))
                    frames.append(gray_resized.astype(np.float32))
                    extracted += 1
                        
                count += 1
            
            cap.release()
            
            if len(frames) < 4:
                return None
            
            # Extract noise residuals
            residuals = []
            for frame in frames:
                denoised = cv2.fastNlMeansDenoising(frame.astype(np.uint8), None, 10, 7, 21)
                denoised = denoised.astype(np.float32)
                residual = frame - denoised
                residual = residual / (np.std(residual) + 1e-6)
                residuals.append(residual)
            
            # Global fingerprint
            global_fingerprint = np.mean(residuals, axis=0)
            global_fingerprint = global_fingerprint / (np.std(global_fingerprint) + 1e-6)
            
            # Per-frame correlations
            correlations = []
            for residual in residuals:
                residual_flat = residual.flatten()
                fingerprint_flat = global_fingerprint.flatten()
                correlation = np.corrcoef(residual_flat, fingerprint_flat)[0, 1]
                
                if np.isnan(correlation):
                    correlation = 0.0
                
                correlations.append(correlation)
            
            correlations = np.array(correlations)
            
            mean_correlation = float(np.mean(correlations))
            std_correlation = float(np.std(correlations))
            positive_threshold = 0.1
            positive_ratio = float(np.sum(correlations > positive_threshold) / len(correlations))
            consistency_score = mean_correlation * positive_ratio / (std_correlation + 0.1)
            consistency_score = float(max(0, min(1, consistency_score)))
            
            return {
                'mean_correlation': mean_correlation,
                'std_correlation': std_correlation,
                'positive_ratio': positive_ratio,
                'consistency_score': consistency_score
            }
            
        except Exception as e:
            print(f"⚠️ PRNU extraction failed: {e}")
            return None
    
    def extract_all_features(self, video_path: str, video_id: str, label: str, source: str) -> VideoFeatures:
        """Extract all features from a video"""
        print(f"🔍 Extracting features from {video_id} ({label}/{source})")
        
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        extraction_notes = []
        
        # Extract trajectory features
        trajectory = self.extract_trajectory_features(video_path)
        if trajectory is None:
            trajectory = {'curvature_mean': 0, 'curvature_std': 0, 'max_curvature': 0, 'mean_distance': 0}
            extraction_notes.append("trajectory_failed")
        
        # Extract optical flow features
        flow = self.extract_optical_flow_features(video_path)
        if flow is None:
            flow = {'jitter_index': 0, 'bg_fg_ratio': 0, 'patch_variance': 0, 'smoothness_score': 0, 'global_mean': 0, 'global_std': 0}
            extraction_notes.append("flow_failed")
        
        # Extract OCR features
        ocr = self.extract_ocr_features(video_path)
        if ocr is None:
            ocr = {'has_text': False, 'char_error_rate': 0, 'frame_stability': 1, 'mutation_rate': 0, 'unique_string_count': 0, 'total_detections': 0}
            extraction_notes.append("ocr_failed")
        
        # Extract PRNU features
        prnu = self.extract_prnu_features(video_path)
        if prnu is None:
            prnu = {'mean_correlation': 0, 'std_correlation': 0, 'positive_ratio': 0, 'consistency_score': 0}
            extraction_notes.append("prnu_failed")
        
        # Create feature object
        features = VideoFeatures(
            video_id=video_id,
            label=label,
            source=source,
            
            # Trajectory
            trajectory_curvature_mean=trajectory['curvature_mean'],
            trajectory_curvature_std=trajectory['curvature_std'],
            trajectory_max_curvature=trajectory['max_curvature'],
            trajectory_mean_distance=trajectory['mean_distance'],
            
            # Optical flow
            flow_jitter_index=flow['jitter_index'],
            flow_bg_fg_ratio=flow['bg_fg_ratio'],
            flow_patch_variance=flow['patch_variance'],
            flow_smoothness_score=flow['smoothness_score'],
            flow_global_mean=flow['global_mean'],
            flow_global_std=flow['global_std'],
            
            # OCR
            ocr_has_text=ocr['has_text'],
            ocr_char_error_rate=ocr['char_error_rate'],
            ocr_frame_stability=ocr['frame_stability'],
            ocr_mutation_rate=ocr['mutation_rate'],
            ocr_unique_string_count=ocr['unique_string_count'],
            ocr_total_detections=ocr['total_detections'],
            
            # PRNU
            prnu_mean_correlation=prnu['mean_correlation'],
            prnu_std_correlation=prnu['std_correlation'],
            prnu_positive_ratio=prnu['positive_ratio'],
            prnu_consistency_score=prnu['consistency_score'],
            
            # Video properties
            duration=duration,
            fps=fps,
            resolution_width=width,
            resolution_height=height,
            
            # Processing metadata
            frames_analyzed=min(16, total_frames),
            extraction_success=len(extraction_notes) == 0,
            extraction_notes=','.join(extraction_notes)
        )
        
        print(f"✅ Features extracted for {video_id}")
        return features

def main():
    """Extract features from calibration dataset"""
    from dataset_collector import CalibrationDatasetCollector
    
    # Load dataset
    collector = CalibrationDatasetCollector()
    if len(collector.videos_metadata) == 0:
        print("❌ No videos in calibration dataset. Run dataset_collector.py first.")
        return
    
    # Extract features
    extractor = FeatureExtractor()
    all_features = []
    
    for video_meta in collector.videos_metadata:
        video_path = collector.dataset_dir / video_meta.label / video_meta.filename
        
        if not video_path.exists():
            print(f"⚠️ Video file not found: {video_path}")
            continue
        
        try:
            features = extractor.extract_all_features(
                str(video_path), 
                video_meta.video_id, 
                video_meta.label, 
                video_meta.source
            )
            all_features.append(features)
        except Exception as e:
            print(f"❌ Failed to extract features from {video_meta.video_id}: {e}")
    
    # Save features to CSV
    if all_features:
        features_df = pd.DataFrame([asdict(f) for f in all_features])
        features_file = collector.dataset_dir / "metadata" / "features.csv"
        features_df.to_csv(features_file, index=False)
        print(f"💾 Saved features for {len(all_features)} videos to {features_file}")
        
        # Print summary
        print(f"\n📊 FEATURE EXTRACTION SUMMARY")
        print(f"Total videos processed: {len(all_features)}")
        print(f"Real videos: {len([f for f in all_features if f.label == 'real'])}")
        print(f"AI videos: {len([f for f in all_features if f.label == 'ai'])}")
        print(f"Successful extractions: {len([f for f in all_features if f.extraction_success])}")
    else:
        print("❌ No features extracted")

if __name__ == "__main__":
    main()
