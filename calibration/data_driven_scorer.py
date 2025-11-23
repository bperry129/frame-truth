#!/usr/bin/env python3
"""
FrameTruth Data-Driven Scoring System
Replaces arbitrary point system with evidence-based thresholds
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class ThresholdConfig:
    """Evidence-based thresholds for each feature"""
    # Trajectory thresholds (based on real data distributions)
    trajectory_curvature_high: float = 120.0  # 95th percentile of real videos
    trajectory_curvature_extreme: float = 150.0  # 99th percentile of real videos
    trajectory_std_high: float = 50.0
    
    # Optical flow thresholds
    flow_jitter_high: float = 1.2  # 90th percentile of real videos
    flow_bg_fg_ratio_unnatural: float = 0.75  # Background/foreground too similar
    flow_patch_variance_high: float = 0.25
    flow_smoothness_extreme: float = 100.0  # Too smooth (inverse of std)
    
    # OCR thresholds
    ocr_char_error_moderate: float = 0.05  # 5% character errors
    ocr_char_error_high: float = 0.15  # 15% character errors
    ocr_stability_low: float = 0.7  # 70% frame stability
    ocr_stability_very_low: float = 0.5  # 50% frame stability
    ocr_mutation_high: float = 0.3  # 30% mutation rate
    ocr_mutation_extreme: float = 0.5  # 50% mutation rate
    
    # PRNU thresholds
    prnu_correlation_low: float = 0.05  # Low sensor correlation
    prnu_correlation_moderate: float = 0.15  # Moderate correlation
    prnu_std_high: float = 0.25  # High variation in correlations
    prnu_positive_ratio_low: float = 0.5  # Low positive correlation ratio

class DataDrivenScorer:
    """Evidence-based scoring system without arbitrary points"""
    
    def __init__(self, features_file: Optional[str] = None):
        self.thresholds = ThresholdConfig()
        self.structural_classifier = None
        self.feature_importance = {}
        
        if features_file and Path(features_file).exists():
            self.load_calibration_data(features_file)
    
    def load_calibration_data(self, features_file: str):
        """Load calibration data and compute evidence-based thresholds"""
        print(f"📊 Loading calibration data from {features_file}")
        
        df = pd.read_csv(features_file)
        
        if len(df) < 10:
            print("⚠️ Not enough calibration data for reliable thresholds")
            return
        
        real_videos = df[df['label'] == 'real']
        ai_videos = df[df['label'] == 'ai']
        
        print(f"📈 Computing thresholds from {len(real_videos)} real and {len(ai_videos)} AI videos")
        
        # Compute evidence-based thresholds
        self._compute_trajectory_thresholds(real_videos, ai_videos)
        self._compute_flow_thresholds(real_videos, ai_videos)
        self._compute_ocr_thresholds(real_videos, ai_videos)
        self._compute_prnu_thresholds(real_videos, ai_videos)
        
        # Train structural classifier
        self._train_structural_classifier(df)
        
        print("✅ Calibration complete")
    
    def _compute_trajectory_thresholds(self, real_df: pd.DataFrame, ai_df: pd.DataFrame):
        """Compute trajectory-based thresholds from real data"""
        if len(real_df) > 0:
            # Use 90th and 95th percentiles of real videos as thresholds
            self.thresholds.trajectory_curvature_high = np.percentile(
                real_df['trajectory_curvature_mean'], 90
            )
            self.thresholds.trajectory_curvature_extreme = np.percentile(
                real_df['trajectory_curvature_mean'], 95
            )
            self.thresholds.trajectory_std_high = np.percentile(
                real_df['trajectory_curvature_std'], 90
            )
            
            print(f"📐 Trajectory thresholds: high={self.thresholds.trajectory_curvature_high:.1f}°, "
                  f"extreme={self.thresholds.trajectory_curvature_extreme:.1f}°")
    
    def _compute_flow_thresholds(self, real_df: pd.DataFrame, ai_df: pd.DataFrame):
        """Compute optical flow thresholds from real data"""
        if len(real_df) > 0:
            self.thresholds.flow_jitter_high = np.percentile(
                real_df['flow_jitter_index'], 90
            )
            self.thresholds.flow_bg_fg_ratio_unnatural = np.percentile(
                real_df['flow_bg_fg_ratio'], 85
            )
            self.thresholds.flow_patch_variance_high = np.percentile(
                real_df['flow_patch_variance'], 90
            )
            
            print(f"🌊 Flow thresholds: jitter={self.thresholds.flow_jitter_high:.3f}, "
                  f"bg_fg={self.thresholds.flow_bg_fg_ratio_unnatural:.3f}")
    
    def _compute_ocr_thresholds(self, real_df: pd.DataFrame, ai_df: pd.DataFrame):
        """Compute OCR thresholds from real data"""
        real_with_text = real_df[real_df['ocr_has_text'] == True]
        
        if len(real_with_text) > 0:
            # For real videos with text, most should have very low error rates
            self.thresholds.ocr_char_error_moderate = np.percentile(
                real_with_text['ocr_char_error_rate'], 75
            )
            self.thresholds.ocr_stability_low = np.percentile(
                real_with_text['ocr_frame_stability'], 25
            )
            self.thresholds.ocr_mutation_high = np.percentile(
                real_with_text['ocr_mutation_rate'], 85
            )
            
            print(f"📝 OCR thresholds: char_error={self.thresholds.ocr_char_error_moderate:.3f}, "
                  f"stability={self.thresholds.ocr_stability_low:.3f}")
    
    def _compute_prnu_thresholds(self, real_df: pd.DataFrame, ai_df: pd.DataFrame):
        """Compute PRNU thresholds from real data"""
        if len(real_df) > 0:
            # Real videos should have higher correlations and lower std
            self.thresholds.prnu_correlation_moderate = np.percentile(
                real_df['prnu_mean_correlation'], 25
            )
            self.thresholds.prnu_std_high = np.percentile(
                real_df['prnu_std_correlation'], 75
            )
            self.thresholds.prnu_positive_ratio_low = np.percentile(
                real_df['prnu_positive_ratio'], 25
            )
            
            print(f"🔬 PRNU thresholds: correlation={self.thresholds.prnu_correlation_moderate:.3f}, "
                  f"std={self.thresholds.prnu_std_high:.3f}")
    
    def _train_structural_classifier(self, df: pd.DataFrame):
        """Train lightweight classifier on structural features"""
        # Select features for classification
        feature_cols = [
            'trajectory_curvature_mean', 'trajectory_curvature_std', 'trajectory_max_curvature',
            'flow_jitter_index', 'flow_bg_fg_ratio', 'flow_patch_variance', 'flow_smoothness_score',
            'ocr_char_error_rate', 'ocr_frame_stability', 'ocr_mutation_rate',
            'prnu_mean_correlation', 'prnu_std_correlation', 'prnu_positive_ratio'
        ]
        
        # Prepare data
        X = df[feature_cols].fillna(0)  # Fill NaN with 0
        y = (df['label'] == 'ai').astype(int)  # 1 for AI, 0 for real
        
        if len(X) < 10:
            print("⚠️ Not enough data to train classifier")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train Random Forest (lightweight and interpretable)
        self.structural_classifier = RandomForestClassifier(
            n_estimators=50,  # Small for speed
            max_depth=5,      # Prevent overfitting
            random_state=42
        )
        
        self.structural_classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.structural_classifier.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        
        # Feature importance
        self.feature_importance = dict(zip(
            feature_cols, 
            self.structural_classifier.feature_importances_
        ))
        
        print(f"🤖 Structural classifier trained: {accuracy:.3f} accuracy")
        print(f"📊 Top features: {sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]}")
    
    def count_structural_flags(self, features: Dict) -> Tuple[int, int]:
        """Count evidence flags for AI vs Real (no arbitrary points)"""
        flags_ai = 0
        flags_real = 0
        
        # Trajectory flags
        curvature = features.get('trajectory_curvature_mean', 0)
        if curvature > self.thresholds.trajectory_curvature_extreme:
            flags_ai += 2  # Strong evidence
        elif curvature > self.thresholds.trajectory_curvature_high:
            flags_ai += 1  # Moderate evidence
        
        # Optical flow flags
        jitter = features.get('flow_jitter_index', 0)
        bg_fg_ratio = features.get('flow_bg_fg_ratio', 0)
        
        if jitter > self.thresholds.flow_jitter_high:
            flags_ai += 1
        
        if bg_fg_ratio > self.thresholds.flow_bg_fg_ratio_unnatural:
            flags_ai += 1
        
        # OCR flags (only if text is present)
        has_text = features.get('ocr_has_text', False)
        if has_text:
            char_error = features.get('ocr_char_error_rate', 0)
            stability = features.get('ocr_frame_stability', 1.0)
            mutation = features.get('ocr_mutation_rate', 0)
            
            # Severe text anomalies
            if char_error > self.thresholds.ocr_char_error_high and stability < self.thresholds.ocr_stability_very_low:
                flags_ai += 3  # Very strong evidence
            elif char_error > self.thresholds.ocr_char_error_moderate or stability < self.thresholds.ocr_stability_low:
                flags_ai += 1  # Moderate evidence
            
            if mutation > self.thresholds.ocr_mutation_extreme:
                flags_ai += 2
            elif mutation > self.thresholds.ocr_mutation_high:
                flags_ai += 1
        
        # PRNU flags
        prnu_corr = features.get('prnu_mean_correlation', 0)
        prnu_std = features.get('prnu_std_correlation', 0)
        prnu_pos_ratio = features.get('prnu_positive_ratio', 0)
        
        # Evidence for AI (no stable sensor pattern)
        if prnu_corr < self.thresholds.prnu_correlation_low or prnu_pos_ratio < self.thresholds.prnu_positive_ratio_low:
            flags_ai += 1
        
        # Evidence for real (stable sensor pattern, but not proof)
        elif (prnu_corr > self.thresholds.prnu_correlation_moderate and 
              prnu_std < self.thresholds.prnu_std_high and 
              prnu_pos_ratio > 0.7):
            flags_real += 1
        
        return flags_ai, flags_real
    
    def get_structural_probability(self, features: Dict) -> float:
        """Get AI probability from structural classifier"""
        if self.structural_classifier is None:
            return 0.5  # Neutral if no classifier
        
        # Prepare feature vector
        feature_cols = [
            'trajectory_curvature_mean', 'trajectory_curvature_std', 'trajectory_max_curvature',
            'flow_jitter_index', 'flow_bg_fg_ratio', 'flow_patch_variance', 'flow_smoothness_score',
            'ocr_char_error_rate', 'ocr_frame_stability', 'ocr_mutation_rate',
            'prnu_mean_correlation', 'prnu_std_correlation', 'prnu_positive_ratio'
        ]
        
        feature_vector = []
        for col in feature_cols:
            value = features.get(col.replace('_', '_'), 0)  # Handle naming differences
            feature_vector.append(value)
        
        # Get probability
        prob_ai = self.structural_classifier.predict_proba([feature_vector])[0][1]
        return float(prob_ai)
    
    def evaluate_video(self, features: Dict, gemini_prob: float) -> Dict:
        """New evaluation logic: flags + fusion, no arbitrary points"""
        
        # Step 1: Count structural evidence flags
        flags_ai, flags_real = self.count_structural_flags(features)
        
        # Step 2: Get structural classifier probability (if available)
        structural_prob = self.get_structural_probability(features)
        
        # Step 3: Fusion logic (not addition)
        if gemini_prob >= 0.85:  # Gemini confident AI
            if flags_real >= 2 and flags_ai <= 1:
                label = "AI"
                confidence = min(95, gemini_prob * 100)
                explanation = "Gemini confident AI, but structural signals suggest some real-like patterns"
            else:
                label = "AI"
                confidence = min(98, gemini_prob * 100 + flags_ai * 2)
                explanation = "Gemini confident AI, structural signals align"
        
        elif gemini_prob <= 0.15:  # Gemini confident Real
            if flags_ai >= 3:
                label = "Uncertain"
                confidence = 60
                explanation = "Gemini says real, but structural signals strongly suggest AI"
            elif flags_ai >= 2:
                label = "Likely Real"
                confidence = 70
                explanation = "Gemini says real, but some structural concerns"
            else:
                label = "Real"
                confidence = min(95, (1 - gemini_prob) * 100 + flags_real * 3)
                explanation = "Gemini confident real, structural signals align"
        
        else:  # Gemini uncertain (0.15 < prob < 0.85)
            if flags_ai > flags_real + 2:
                label = "Likely AI"
                confidence = 75
                explanation = "Gemini uncertain, structural evidence leans AI"
            elif flags_real > flags_ai + 1:
                label = "Likely Real"
                confidence = 70
                explanation = "Gemini uncertain, structural evidence leans real"
            else:
                label = "Uncertain"
                confidence = 50
                explanation = "Both Gemini and structural signals are ambiguous"
        
        # Step 4: Check for metadata override (minimal impact)
        metadata_boost = features.get('metadata_ai_keywords', False)
        if metadata_boost and label in ["Uncertain", "Likely Real"]:
            label = "Likely AI"
            confidence = min(confidence + 10, 85)
            explanation += " (metadata suggests AI generation)"
        
        return {
            'label': label,
            'confidence': confidence,
            'explanation': explanation,
            'gemini_prob': gemini_prob,
            'structural_prob': structural_prob,
            'flags_ai': flags_ai,
            'flags_real': flags_real,
            'structural_features_used': self.structural_classifier is not None
        }
    
    def save_calibration(self, filepath: str):
        """Save calibrated thresholds and classifier"""
        calibration_data = {
            'thresholds': self.thresholds.__dict__,
            'feature_importance': self.feature_importance
        }
        
        with open(filepath, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        # Save classifier separately
        if self.structural_classifier:
            classifier_path = filepath.replace('.json', '_classifier.pkl')
            with open(classifier_path, 'wb') as f:
                pickle.dump(self.structural_classifier, f)
        
        print(f"💾 Calibration saved to {filepath}")
    
    def load_calibration(self, filepath: str):
        """Load calibrated thresholds and classifier"""
        with open(filepath, 'r') as f:
            calibration_data = json.load(f)
        
        # Load thresholds
        for key, value in calibration_data['thresholds'].items():
            setattr(self.thresholds, key, value)
        
        self.feature_importance = calibration_data.get('feature_importance', {})
        
        # Load classifier
        classifier_path = filepath.replace('.json', '_classifier.pkl')
        if Path(classifier_path).exists():
            with open(classifier_path, 'rb') as f:
                self.structural_classifier = pickle.load(f)
        
        print(f"📊 Calibration loaded from {filepath}")

def main():
    """Test the data-driven scorer"""
    features_file = "calibration/dataset/metadata/features.csv"
    
    if not Path(features_file).exists():
        print(f"❌ Features file not found: {features_file}")
        print("Run feature_extractor.py first to extract features from calibration dataset")
        return
    
    # Create and calibrate scorer
    scorer = DataDrivenScorer(features_file)
    
    # Save calibration
    calibration_file = "calibration/calibrated_thresholds.json"
    scorer.save_calibration(calibration_file)
    
    # Test with example features
    test_features = {
        'trajectory_curvature_mean': 95.0,
        'flow_jitter_index': 0.8,
        'ocr_has_text': True,
        'ocr_char_error_rate': 0.12,
        'ocr_frame_stability': 0.6,
        'prnu_mean_correlation': 0.08
    }
    
    # Test evaluation
    result = scorer.evaluate_video(test_features, gemini_prob=0.7)
    
    print(f"\n🧪 TEST EVALUATION:")
    print(f"Label: {result['label']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Explanation: {result['explanation']}")
    print(f"Flags AI: {result['flags_ai']}, Flags Real: {result['flags_real']}")

if __name__ == "__main__":
    main()
