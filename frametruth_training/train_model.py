"""
FrameTruth Model Training Script
Train evidence-based AI detection models on forensic features
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def train_initial_model():
    """
    Train the first evidence-based model on 87 videos
    Replaces arbitrary point system with learned weights
    """
    
    print("🎯 FrameTruth Evidence-Based Model Training")
    print("=" * 50)
    
    # 1. Load your dataset
    print("📊 Loading dataset...")
    data_path = Path('data/dataset.csv')
    if not data_path.exists():
        # Try alternative path if running from parent directory
        data_path = Path('frametruth_training/data/dataset.csv')
        if not data_path.exists():
            print(f"❌ Dataset not found at data/dataset.csv or frametruth_training/data/dataset.csv")
            print("Please ensure dataset.csv exists in the data/ directory")
            return None
        else:
            print(f"✅ Found dataset at {data_path}")
    else:
        print(f"✅ Found dataset at {data_path}")
    
    data = pd.read_csv(data_path)
    print(f"✅ Loaded {len(data)} videos")
    
    # 2. Check label distribution
    label_counts = data['label'].value_counts()
    print(f"📋 Label distribution:")
    for label, count in label_counts.items():
        print(f"   {label}: {count} videos ({count/len(data)*100:.1f}%)")
    
    # 3. Remove constant features (ChatGPT identified these as useless)
    drop_features = [
        'video_id', 'video_path', 'label',  # Non-features
        'prnu_positive_ratio',              # Always 1.0
        'prnu_consistency_score',           # Always 1.0
        'flow_bg_fg_ratio',                # Single value
        'ocr_char_error_rate',             # Single value
        'freq_spectrum_slope_mean'         # Single value
    ]
    
    # Check which features actually exist
    available_features = [col for col in drop_features if col in data.columns]
    missing_features = [col for col in drop_features if col not in data.columns]
    
    if missing_features:
        print(f"⚠️ Some expected features not found: {missing_features}")
    
    X = data.drop(columns=available_features, errors='ignore')
    y = data['label']
    
    print(f"🔬 Training features: {len(X.columns)}")
    print(f"📊 Training samples: {len(X)}")
    
    # 4. Handle any missing values
    if X.isnull().sum().sum() > 0:
        print("⚠️ Found missing values, filling with median...")
        X = X.fillna(X.median())
    
    # 5. Split for validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"📊 Training set: {len(X_train)} videos")
    print(f"📊 Test set: {len(X_test)} videos")
    
    # 6. Scale features
    print("⚖️ Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 7. Train ensemble models
    print("\n🤖 Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    
    print("🤖 Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    gb_model.fit(X_train_scaled, y_train)
    
    # 8. Evaluate performance
    print("\n📊 EVALUATING MODEL PERFORMANCE")
    print("-" * 40)
    
    rf_accuracy = rf_model.score(X_test_scaled, y_test)
    gb_accuracy = gb_model.score(X_test_scaled, y_test)
    ensemble_accuracy = (rf_accuracy + gb_accuracy) / 2
    
    print(f"Random Forest Accuracy: {rf_accuracy:.1%}")
    print(f"Gradient Boosting Accuracy: {gb_accuracy:.1%}")
    print(f"Ensemble Average: {ensemble_accuracy:.1%}")
    
    # Cross-validation for more robust estimate
    print("\n🔄 Cross-validation scores:")
    rf_cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
    gb_cv_scores = cross_val_score(gb_model, X_train_scaled, y_train, cv=5)
    
    print(f"Random Forest CV: {rf_cv_scores.mean():.1%} (+/- {rf_cv_scores.std() * 2:.1%})")
    print(f"Gradient Boosting CV: {gb_cv_scores.mean():.1%} (+/- {gb_cv_scores.std() * 2:.1%})")
    
    # 9. Detailed classification report
    print("\n📋 DETAILED CLASSIFICATION REPORT")
    print("-" * 40)
    
    rf_pred = rf_model.predict(X_test_scaled)
    gb_pred = gb_model.predict(X_test_scaled)
    
    print("Random Forest:")
    print(classification_report(y_test, rf_pred))
    
    print("Gradient Boosting:")
    print(classification_report(y_test, gb_pred))
    
    # 10. Feature importance analysis
    print("\n🎯 FEATURE IMPORTANCE ANALYSIS")
    print("-" * 40)
    
    feature_importance = {}
    for i, feature in enumerate(X.columns):
        rf_imp = rf_model.feature_importances_[i]
        gb_imp = gb_model.feature_importances_[i]
        feature_importance[feature] = (rf_imp + gb_imp) / 2
    
    # Sort by importance
    sorted_features = sorted(feature_importance.items(), 
                           key=lambda x: x[1], reverse=True)
    
    print("TOP 15 MOST IMPORTANT FEATURES:")
    for i, (feature, importance) in enumerate(sorted_features[:15], 1):
        print(f"{i:2d}. {feature:<30} {importance:.3f}")
    
    # 11. Save models and metadata
    print("\n💾 SAVING MODELS")
    print("-" * 40)
    
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Save models
    joblib.dump(rf_model, models_dir / 'random_forest_v1.pkl')
    joblib.dump(gb_model, models_dir / 'gradient_boosting_v1.pkl')
    joblib.dump(scaler, models_dir / 'feature_scaler_v1.pkl')
    
    print("✅ Saved random_forest_v1.pkl")
    print("✅ Saved gradient_boosting_v1.pkl")
    print("✅ Saved feature_scaler_v1.pkl")
    
    # Save feature names and importance
    model_metadata = {
        'version': 1,
        'training_samples': len(X),
        'ai_samples': len(data[data['label'] == 'AI']),
        'real_samples': len(data[data['label'] == 'real']),
        'features': list(X.columns),
        'feature_importance': dict(sorted_features),
        'accuracy': {
            'random_forest': float(rf_accuracy),
            'gradient_boosting': float(gb_accuracy),
            'ensemble': float(ensemble_accuracy),
            'rf_cv_mean': float(rf_cv_scores.mean()),
            'gb_cv_mean': float(gb_cv_scores.mean())
        },
        'trained_date': datetime.now().isoformat(),
        'training_config': {
            'test_size': 0.2,
            'random_state': 42,
            'rf_n_estimators': 100,
            'gb_n_estimators': 100
        }
    }
    
    with open(models_dir / 'model_metadata_v1.json', 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    print("✅ Saved model_metadata_v1.json")
    
    # 12. Summary
    print(f"\n🎉 TRAINING COMPLETE!")
    print("=" * 50)
    print(f"📊 Dataset: {len(X)} videos ({len(data[data['label'] == 'AI'])} AI, {len(data[data['label'] == 'real'])} Real)")
    print(f"🎯 Accuracy: {ensemble_accuracy:.1%} (Random Forest: {rf_accuracy:.1%}, Gradient Boosting: {gb_accuracy:.1%})")
    print(f"🔬 Features: {len(X.columns)} forensic features")
    print(f"💾 Models saved to: {models_dir.absolute()}")
    print(f"\n🚀 Ready to deploy evidence-based AI detector!")
    print(f"📈 This replaces your arbitrary +40/+35/+30 point system")
    print(f"🧠 with learned weights from {len(X)} real-world videos!")
    
    return rf_model, gb_model, scaler, model_metadata

def test_model_prediction():
    """Test the trained model with a sample prediction"""
    
    print("\n🧪 TESTING MODEL PREDICTION")
    print("-" * 40)
    
    models_dir = Path('models')
    
    # Check if models exist
    if not (models_dir / 'random_forest_v1.pkl').exists():
        print("❌ Models not found. Run training first.")
        return
    
    # Load models
    rf_model = joblib.load(models_dir / 'random_forest_v1.pkl')
    gb_model = joblib.load(models_dir / 'gradient_boosting_v1.pkl')
    scaler = joblib.load(models_dir / 'feature_scaler_v1.pkl')
    
    with open(models_dir / 'model_metadata_v1.json') as f:
        metadata = json.load(f)
    
    print(f"✅ Loaded models (accuracy: {metadata['accuracy']['ensemble']:.1%})")
    
    # Load a sample from the dataset
    # Use the same path logic as in training
    data_path = Path('data/dataset.csv')
    if not data_path.exists():
        data_path = Path('frametruth_training/data/dataset.csv')
    
    data = pd.read_csv(data_path)
    sample = data.iloc[0]  # First video
    
    # Extract features
    feature_names = metadata['features']
    feature_vector = []
    for feature_name in feature_names:
        if feature_name in sample:
            feature_vector.append(sample[feature_name])
        else:
            feature_vector.append(0.0)
    
    # Scale and predict
    feature_vector = np.array(feature_vector).reshape(1, -1)
    feature_vector_scaled = scaler.transform(feature_vector)
    
    rf_prob = rf_model.predict_proba(feature_vector_scaled)[0][1]  # AI probability
    gb_prob = gb_model.predict_proba(feature_vector_scaled)[0][1]  # AI probability
    ensemble_prob = (rf_prob + gb_prob) / 2
    
    print(f"\n📊 Sample Prediction:")
    print(f"Video: {sample['video_id']}")
    print(f"True Label: {sample['label']}")
    print(f"Random Forest: {rf_prob:.1%} AI probability")
    print(f"Gradient Boosting: {gb_prob:.1%} AI probability")
    print(f"Ensemble: {ensemble_prob:.1%} AI probability")
    
    if ensemble_prob >= 0.8:
        verdict = "AI Generated (High Confidence)"
    elif ensemble_prob <= 0.2:
        verdict = "Real Video (High Confidence)"
    else:
        verdict = "Uncertain (Low Confidence)"
    
    print(f"Verdict: {verdict}")

if __name__ == "__main__":
    # Train the model
    result = train_initial_model()
    
    if result is not None:
        # Test with a sample prediction
        test_model_prediction()
        
        print(f"\n🎯 NEXT STEPS:")
        print("1. Review the feature importance rankings")
        print("2. Update your backend to use the trained models")
        print("3. Deploy the evidence-based scorer")
        print("4. Monitor performance on real users")
        print("5. Start collecting admin feedback for continuous learning")
