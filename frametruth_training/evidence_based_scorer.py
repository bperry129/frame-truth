"""
Evidence-Based AI Detection Scorer
Replaces arbitrary point system with trained machine learning models
"""

import joblib
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple

class EvidenceBasedScorer:
    """
    Evidence-based AI detection using trained machine learning models
    Replaces the arbitrary +40/+35/+30 point system with learned weights
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the evidence-based scorer
        
        Args:
            models_dir: Directory containing the trained models
        """
        self.models_dir = Path(models_dir)
        self.rf_model = None
        self.gb_model = None
        self.scaler = None
        self.metadata = None
        self.feature_names = None
        
        self._load_models()
    
    def _load_models(self):
        """Load the trained models and metadata"""
        try:
            # Load models
            self.rf_model = joblib.load(self.models_dir / 'random_forest_v1.pkl')
            self.gb_model = joblib.load(self.models_dir / 'gradient_boosting_v1.pkl')
            self.scaler = joblib.load(self.models_dir / 'feature_scaler_v1.pkl')
            
            # Load metadata
            with open(self.models_dir / 'model_metadata_v1.json') as f:
                self.metadata = json.load(f)
            
            self.feature_names = self.metadata['features']
            
            print(f"🤖 Loaded evidence-based models:")
            print(f"   📊 Trained on {self.metadata['training_samples']} videos")
            print(f"   🎯 Accuracy: {self.metadata['accuracy']['ensemble']:.1%}")
            print(f"   📅 Trained: {self.metadata['trained_date'][:10]}")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("Please ensure models are trained first by running train_model.py")
            raise
    
    def calculate_ai_probability(self, features_dict: Dict) -> float:
        """
        Calculate AI probability using trained models
        
        Args:
            features_dict: Dictionary with all forensic features
            
        Returns:
            float: AI probability (0-100%)
        """
        if self.rf_model is None or self.gb_model is None:
            raise ValueError("Models not loaded. Please check model files.")
        
        # 1. Extract features in correct order
        feature_vector = []
        missing_features = []
        
        for feature_name in self.feature_names:
            if feature_name in features_dict:
                feature_vector.append(features_dict[feature_name])
            else:
                feature_vector.append(0.0)  # Default for missing features
                missing_features.append(feature_name)
        
        if missing_features:
            print(f"⚠️ Missing features (using 0.0): {missing_features[:5]}...")
        
        # 2. Scale features
        feature_vector = np.array(feature_vector).reshape(1, -1)
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # 3. Get predictions from both models
        rf_prob = self.rf_model.predict_proba(feature_vector_scaled)[0][1]  # AI probability
        gb_prob = self.gb_model.predict_proba(feature_vector_scaled)[0][1]  # AI probability
        
        # 4. Ensemble average
        ensemble_prob = (rf_prob + gb_prob) / 2
        
        return ensemble_prob * 100  # Convert to percentage
    
    def get_feature_contributions(self, features_dict: Dict) -> List[Tuple[str, float]]:
        """
        Show which features contributed most to the decision
        
        Args:
            features_dict: Dictionary with all forensic features
            
        Returns:
            List of (feature_name, contribution_score) tuples
        """
        if self.metadata is None:
            return []
        
        # Get feature importance from metadata
        contributions = []
        for feature_name in self.feature_names:
            importance = self.metadata['feature_importance'][feature_name]
            value = features_dict.get(feature_name, 0.0)
            
            # Normalize value (simple approach)
            normalized_value = min(abs(value) / 1000.0, 1.0)  # Cap at 1.0
            contribution = importance * normalized_value
            
            contributions.append((feature_name, contribution))
        
        # Return top 5 contributors
        contributions.sort(key=lambda x: x[1], reverse=True)
        return contributions[:5]
    
    def get_verdict(self, ai_probability: float) -> Tuple[str, str]:
        """
        Get human-readable verdict based on AI probability
        
        Args:
            ai_probability: AI probability (0-100%)
            
        Returns:
            Tuple of (verdict, confidence)
        """
        if ai_probability >= 80:
            return "AI Generated", "High"
        elif ai_probability <= 20:
            return "Real Video", "High"
        elif ai_probability >= 60:
            return "Likely AI", "Medium"
        elif ai_probability <= 40:
            return "Likely Real", "Medium"
        else:
            return "Uncertain", "Low"
    
    def analyze_video_evidence_based(self, features_dict: Dict) -> Dict:
        """
        Complete evidence-based analysis replacing the old point system
        
        Args:
            features_dict: Dictionary with all forensic features
            
        Returns:
            Dictionary with analysis results
        """
        # Calculate AI probability
        ai_probability = self.calculate_ai_probability(features_dict)
        
        # Get verdict
        verdict, confidence = self.get_verdict(ai_probability)
        
        # Get top contributing features
        top_contributors = self.get_feature_contributions(features_dict)
        
        # Format results
        result = {
            'ai_probability': round(ai_probability, 1),
            'verdict': verdict,
            'confidence': confidence,
            'model_version': self.metadata['version'],
            'model_accuracy': f"{self.metadata['accuracy']['ensemble']:.1%}",
            'top_contributors': [
                {
                    'feature': feature,
                    'contribution': round(contribution, 3),
                    'description': self._get_feature_description(feature)
                }
                for feature, contribution in top_contributors
            ],
            'analysis_method': 'Evidence-Based Machine Learning',
            'training_data': f"{self.metadata['training_samples']} videos"
        }
        
        return result
    
    def _get_feature_description(self, feature_name: str) -> str:
        """Get human-readable description of a feature"""
        descriptions = {
            'ocr_frame_stability': 'Text consistency across frames',
            'trajectory_curvature_mean': 'Motion path complexity',
            'flow_global_mean': 'Overall motion magnitude',
            'flow_smoothness_score': 'Motion smoothness quality',
            'ocr_total_detections': 'Amount of text detected',
            'ocr_unique_string_count': 'Variety of text content',
            'noise_variance_g_mean': 'Green channel noise patterns',
            'ocr_mutation_rate': 'Text change frequency',
            'flow_global_std': 'Motion consistency',
            'flow_patch_variance': 'Local motion variation',
            'cross_channel_corr_rg_mean': 'Color channel correlation',
            'spatial_autocorr_mean': 'Texture repetitiveness',
            'trajectory_curvature_std': 'Motion path variation',
            'avg_bitrate': 'Video compression rate',
            'noise_variance_b_mean': 'Blue channel noise patterns'
        }
        
        return descriptions.get(feature_name, feature_name.replace('_', ' ').title())

def main():
    """Test the evidence-based scorer"""
    print("🧪 Testing Evidence-Based Scorer")
    print("=" * 40)
    
    try:
        # Initialize scorer
        scorer = EvidenceBasedScorer()
        
        # Load a sample from the dataset for testing
        data_path = Path('data/dataset.csv')
        if not data_path.exists():
            data_path = Path('frametruth_training/data/dataset.csv')
        
        if data_path.exists():
            data = pd.read_csv(data_path)
            sample = data.iloc[0]  # First video
            
            # Convert to features dictionary
            features_dict = sample.to_dict()
            
            # Analyze with evidence-based scorer
            result = scorer.analyze_video_evidence_based(features_dict)
            
            print(f"\n📊 Sample Analysis:")
            print(f"Video: {sample['video_id']}")
            print(f"True Label: {sample['label']}")
            print(f"AI Probability: {result['ai_probability']:.1f}%")
            print(f"Verdict: {result['verdict']} ({result['confidence']} Confidence)")
            print(f"Model Accuracy: {result['model_accuracy']}")
            
            print(f"\n🎯 Top Contributing Features:")
            for contrib in result['top_contributors']:
                print(f"  • {contrib['description']}: {contrib['contribution']:.3f}")
            
            print(f"\n✅ Evidence-based scorer working correctly!")
            
        else:
            print("❌ Dataset not found for testing")
            
    except Exception as e:
        print(f"❌ Error testing scorer: {e}")

if __name__ == "__main__":
    main()
