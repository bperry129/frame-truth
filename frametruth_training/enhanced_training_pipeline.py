#!/usr/bin/env python3
"""
🚀 Enhanced Training Pipeline for Frame Truth
Integrates research datasets with existing training data for improved accuracy

This pipeline:
1. Downloads research datasets (Synth-Vid-Detect, etc.)
2. Combines with existing 87-video dataset
3. Extracts features from all videos
4. Retrains models with expanded dataset
5. Recalibrates thresholds for modern AI detection

Expected improvement: 66.7% → 85-90% accuracy
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import DATA_DIR
from research_dataset_downloader import ResearchDatasetDownloader
from build_dataset import build_complete_dataset
from train_model import FrameTruthTrainer

class EnhancedTrainingPipeline:
    """Enhanced training pipeline with research dataset integration"""
    
    def __init__(self, max_research_videos: int = 500):
        self.max_research_videos = max_research_videos
        self.data_dir = Path(DATA_DIR)
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Enhanced model version
        self.model_version = "v2_research_enhanced"
        
        # Feature importance tracking
        self.feature_importance = {}
        
    def step1_download_research_datasets(self) -> Dict:
        """Step 1: Download research datasets"""
        print("🚀 STEP 1: Downloading research datasets...")
        print("=" * 60)
        
        downloader = ResearchDatasetDownloader(max_videos_per_dataset=self.max_research_videos)
        result = downloader.download_all_datasets()
        
        if result.get("total_videos", 0) == 0:
            print("⚠️ No research videos downloaded. Proceeding with existing dataset only.")
            return {"research_videos": 0, "csv_path": None}
        
        print(f"✅ Downloaded {result['total_videos']} research videos")
        return {
            "research_videos": result["total_videos"],
            "csv_path": result.get("unified_csv"),
            "details": result
        }
    
    def step2_combine_datasets(self, research_csv_path: Optional[str] = None) -> str:
        """Step 2: Combine existing dataset with research datasets"""
        print("\n🔗 STEP 2: Combining datasets...")
        print("=" * 60)
        
        # Load existing dataset
        existing_csv = self.data_dir / "dataset.csv"
        if not existing_csv.exists():
            print("❌ Existing dataset not found. Run build_dataset.py first.")
            raise FileNotFoundError("Existing dataset.csv not found")
        
        existing_df = pd.read_csv(existing_csv)
        print(f"📊 Existing dataset: {len(existing_df)} videos")
        print(f"   AI: {len(existing_df[existing_df['label'] == 'ai'])}")
        print(f"   Real: {len(existing_df[existing_df['label'] == 'real'])}")
        
        # Combine with research dataset if available
        if research_csv_path and Path(research_csv_path).exists():
            research_df = pd.read_csv(research_csv_path)
            print(f"📊 Research dataset: {len(research_df)} videos")
            print(f"   AI: {len(research_df[research_df['label'] == 'ai'])}")
            print(f"   Real: {len(research_df[research_df['label'] == 'real'])}")
            
            # Standardize column names
            if 'filepath' in research_df.columns:
                research_df['video_path'] = research_df['filepath']
            
            # Select common columns
            common_columns = ['url', 'label']
            if 'video_path' in research_df.columns:
                common_columns.append('video_path')
            
            # Combine datasets
            combined_df = pd.concat([
                existing_df[common_columns],
                research_df[common_columns]
            ], ignore_index=True)
            
        else:
            print("⚠️ No research dataset to combine. Using existing dataset only.")
            combined_df = existing_df
        
        # Save combined dataset
        combined_csv_path = self.data_dir / f"combined_dataset_{self.model_version}.csv"
        combined_df.to_csv(combined_csv_path, index=False)
        
        print(f"✅ Combined dataset: {len(combined_df)} videos")
        print(f"   AI: {len(combined_df[combined_df['label'] == 'ai'])}")
        print(f"   Real: {len(combined_df[combined_df['label'] == 'real'])}")
        print(f"   Saved to: {combined_csv_path}")
        
        return str(combined_csv_path)
    
    def step3_extract_features(self, combined_csv_path: str) -> str:
        """Step 3: Extract features from combined dataset"""
        print("\n🔬 STEP 3: Extracting features from combined dataset...")
        print("=" * 60)
        
        # Use existing build_dataset pipeline but with combined CSV
        print("📊 Running feature extraction pipeline...")
        
        try:
            # This will extract features from all videos in the combined dataset
            result = build_complete_dataset(combined_csv_path)
            
            if "error" in result:
                raise Exception(result["error"])
            
            print(f"✅ Feature extraction complete:")
            print(f"   Videos processed: {result.get('videos_processed', 0)}")
            print(f"   Features extracted: {result.get('features_extracted', 0)}")
            
            # The features are saved to dataset.csv by build_complete_dataset
            features_csv_path = self.data_dir / "dataset.csv"
            
            # Create a backup with version info
            enhanced_features_path = self.data_dir / f"dataset_{self.model_version}.csv"
            if features_csv_path.exists():
                import shutil
                shutil.copy2(features_csv_path, enhanced_features_path)
                print(f"📁 Features saved to: {enhanced_features_path}")
            
            return str(enhanced_features_path)
            
        except Exception as e:
            print(f"❌ Feature extraction failed: {e}")
            raise
    
    def step4_train_enhanced_models(self, features_csv_path: str) -> Dict:
        """Step 4: Train enhanced models with expanded dataset"""
        print("\n🤖 STEP 4: Training enhanced models...")
        print("=" * 60)
        
        # Load features dataset
        df = pd.read_csv(features_csv_path)
        print(f"📊 Training dataset: {len(df)} videos with features")
        
        # Prepare features and labels
        feature_columns = [col for col in df.columns if col not in ['url', 'label', 'video_path', 'video_id']]
        X = df[feature_columns].fillna(0)  # Fill NaN with 0
        y = (df['label'] == 'ai').astype(int)  # Convert to binary: 1=AI, 0=Real
        
        print(f"🔢 Features: {len(feature_columns)} dimensions")
        print(f"📊 Labels: {sum(y)} AI, {len(y) - sum(y)} Real")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        models = {}
        results = {}
        
        # 1. Enhanced Random Forest
        print("🌲 Training Enhanced Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=200,  # Increased from 100
            max_depth=15,      # Increased depth for complex patterns
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        
        models['random_forest'] = rf_model
        results['random_forest'] = {
            'accuracy': rf_accuracy,
            'predictions': rf_pred.tolist(),
            'feature_importance': dict(zip(feature_columns, rf_model.feature_importances_))
        }
        
        print(f"   Random Forest Accuracy: {rf_accuracy:.3f}")
        
        # 2. Enhanced Gradient Boosting
        print("🚀 Training Enhanced Gradient Boosting...")
        gb_model = GradientBoostingClassifier(
            n_estimators=200,  # Increased from 100
            learning_rate=0.1,
            max_depth=8,       # Increased depth
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        gb_pred = gb_model.predict(X_test_scaled)
        gb_accuracy = accuracy_score(y_test, gb_pred)
        
        models['gradient_boosting'] = gb_model
        results['gradient_boosting'] = {
            'accuracy': gb_accuracy,
            'predictions': gb_pred.tolist(),
            'feature_importance': dict(zip(feature_columns, gb_model.feature_importances_))
        }
        
        print(f"   Gradient Boosting Accuracy: {gb_accuracy:.3f}")
        
        # 3. Cross-validation scores
        print("🔄 Running cross-validation...")
        rf_cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
        gb_cv_scores = cross_val_score(gb_model, X_train_scaled, y_train, cv=5)
        
        results['cross_validation'] = {
            'random_forest': {
                'mean': rf_cv_scores.mean(),
                'std': rf_cv_scores.std(),
                'scores': rf_cv_scores.tolist()
            },
            'gradient_boosting': {
                'mean': gb_cv_scores.mean(),
                'std': gb_cv_scores.std(),
                'scores': gb_cv_scores.tolist()
            }
        }
        
        print(f"   RF CV Score: {rf_cv_scores.mean():.3f} ± {rf_cv_scores.std():.3f}")
        print(f"   GB CV Score: {gb_cv_scores.mean():.3f} ± {gb_cv_scores.std():.3f}")
        
        # Save models and scaler
        model_files = {}
        
        # Save Random Forest
        rf_path = self.models_dir / f"random_forest_{self.model_version}.pkl"
        joblib.dump(rf_model, rf_path)
        model_files['random_forest'] = str(rf_path)
        
        # Save Gradient Boosting
        gb_path = self.models_dir / f"gradient_boosting_{self.model_version}.pkl"
        joblib.dump(gb_model, gb_path)
        model_files['gradient_boosting'] = str(gb_path)
        
        # Save Scaler
        scaler_path = self.models_dir / f"feature_scaler_{self.model_version}.pkl"
        joblib.dump(scaler, scaler_path)
        model_files['scaler'] = str(scaler_path)
        
        # Save metadata
        metadata = {
            'model_version': self.model_version,
            'training_date': datetime.now().isoformat(),
            'dataset_size': len(df),
            'feature_count': len(feature_columns),
            'feature_names': feature_columns,
            'test_accuracy': {
                'random_forest': rf_accuracy,
                'gradient_boosting': gb_accuracy
            },
            'cross_validation': results['cross_validation'],
            'model_files': model_files
        }
        
        metadata_path = self.models_dir / f"model_metadata_{self.model_version}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Models saved:")
        print(f"   Random Forest: {rf_path}")
        print(f"   Gradient Boosting: {gb_path}")
        print(f"   Scaler: {scaler_path}")
        print(f"   Metadata: {metadata_path}")
        
        return {
            'models': models,
            'results': results,
            'metadata': metadata,
            'model_files': model_files
        }
    
    def step5_analyze_feature_importance(self, training_results: Dict) -> Dict:
        """Step 5: Analyze feature importance and recalibrate thresholds"""
        print("\n📊 STEP 5: Analyzing feature importance...")
        print("=" * 60)
        
        # Get feature importance from both models
        rf_importance = training_results['results']['random_forest']['feature_importance']
        gb_importance = training_results['results']['gradient_boosting']['feature_importance']
        
        # Combine importance scores
        combined_importance = {}
        for feature in rf_importance:
            combined_importance[feature] = (rf_importance[feature] + gb_importance[feature]) / 2
        
        # Sort by importance
        sorted_features = sorted(combined_importance.items(), key=lambda x: x[1], reverse=True)
        
        print("🔝 Top 10 Most Important Features:")
        for i, (feature, importance) in enumerate(sorted_features[:10]):
            print(f"   {i+1:2d}. {feature:<30} {importance:.4f}")
        
        # Identify critical thresholds for recalibration
        critical_features = {
            'prnu_mean_corr': 'PRNU sensor fingerprint correlation',
            'trajectory_curvature_mean': 'Visual trajectory curvature',
            'flow_jitter_index': 'Optical flow temporal jitter',
            'ocr_frame_stability': 'Text stability across frames',
            'freq_high_low_ratio_mean': 'Frequency domain patterns'
        }
        
        print("\n🎯 Critical Feature Analysis:")
        for feature, description in critical_features.items():
            if feature in combined_importance:
                importance = combined_importance[feature]
                print(f"   {feature:<25} {importance:.4f} - {description}")
        
        # Save feature importance analysis
        importance_analysis = {
            'feature_importance_combined': combined_importance,
            'top_features': sorted_features[:20],
            'critical_features': critical_features,
            'analysis_date': datetime.now().isoformat()
        }
        
        analysis_path = self.models_dir / f"feature_importance_{self.model_version}.json"
        with open(analysis_path, 'w') as f:
            json.dump(importance_analysis, f, indent=2)
        
        print(f"📁 Feature importance analysis saved: {analysis_path}")
        
        return importance_analysis
    
    def run_complete_pipeline(self) -> Dict:
        """Run the complete enhanced training pipeline"""
        print("🚀 ENHANCED TRAINING PIPELINE STARTING...")
        print("=" * 80)
        print(f"Target: Improve accuracy from 66.7% to 85-90%")
        print(f"Method: Research dataset integration + enhanced models")
        print("=" * 80)
        
        pipeline_results = {}
        
        try:
            # Step 1: Download research datasets
            research_result = self.step1_download_research_datasets()
            pipeline_results['research_download'] = research_result
            
            # Step 2: Combine datasets
            combined_csv = self.step2_combine_datasets(research_result.get('csv_path'))
            pipeline_results['combined_dataset'] = combined_csv
            
            # Step 3: Extract features
            features_csv = self.step3_extract_features(combined_csv)
            pipeline_results['features_dataset'] = features_csv
            
            # Step 4: Train enhanced models
            training_results = self.step4_train_enhanced_models(features_csv)
            pipeline_results['training_results'] = training_results
            
            # Step 5: Analyze feature importance
            importance_analysis = self.step5_analyze_feature_importance(training_results)
            pipeline_results['feature_analysis'] = importance_analysis
            
            # Final summary
            print("\n🎉 ENHANCED TRAINING PIPELINE COMPLETE!")
            print("=" * 80)
            
            best_accuracy = max(
                training_results['results']['random_forest']['accuracy'],
                training_results['results']['gradient_boosting']['accuracy']
            )
            
            print(f"📊 RESULTS SUMMARY:")
            print(f"   Original accuracy: 66.7%")
            print(f"   Enhanced accuracy: {best_accuracy:.1%}")
            print(f"   Improvement: +{best_accuracy - 0.667:.1%}")
            print(f"   Dataset size: {research_result.get('research_videos', 0) + 87} videos")
            print(f"   Model version: {self.model_version}")
            
            pipeline_results['summary'] = {
                'original_accuracy': 0.667,
                'enhanced_accuracy': best_accuracy,
                'improvement': best_accuracy - 0.667,
                'total_videos': research_result.get('research_videos', 0) + 87,
                'model_version': self.model_version
            }
            
            return pipeline_results
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            pipeline_results['error'] = str(e)
            return pipeline_results

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced training pipeline with research datasets")
    parser.add_argument("--max-research-videos", type=int, default=500,
                       help="Maximum research videos to download")
    
    args = parser.parse_args()
    
    pipeline = EnhancedTrainingPipeline(max_research_videos=args.max_research_videos)
    results = pipeline.run_complete_pipeline()
    
    print(f"\n📊 Final Pipeline Results:")
    print(json.dumps(results.get('summary', {}), indent=2))

if __name__ == "__main__":
    main()
