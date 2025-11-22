# FrameTruth AI Detection Enhancement Roadmap

## 🎯 **Mission: Transform FrameTruth from "Good" to "Scary Good"**

Based on cutting-edge research and ChatGPT analysis, this roadmap outlines the systematic enhancement of FrameTruth's AI detection capabilities from ~85% accuracy to 95%+ accuracy through ensemble methods, structured analysis, and advanced feature extraction.

---

## 📊 **Current System Analysis**

### ✅ **What We Have:**
- **ReStraV Trajectory Analysis**: Lightweight OpenCV-based visual trajectory curvature detection
- **Gemini Visual Forensics**: Comprehensive frame-by-frame analysis with detailed prompting
- **Metadata Scanning**: AI keyword detection in video titles/descriptions
- **Basic Fusion**: Simple threshold-based score boosting

### ❌ **Critical Gaps:**
- **Single Point of Failure**: Gemini is the sole decision maker
- **Vibes-Based Analysis**: No structured numeric features
- **Inconsistent Outputs**: Flip-flopping between "everything AI" and "everything real"
- **Missing Temporal Signals**: No optical flow or motion consistency analysis
- **Weak Text Analysis**: OCR only happens inside Gemini's "black box"
- **No Ensemble**: All signals feed into one model instead of intelligent fusion

---

## 🚀 **Three-Phase Enhancement Plan**

---

# **PHASE 1: Core Infrastructure (High Impact - 2 weeks)**

## **Goal**: Add fundamental missing signals and fix current instability issues
**Expected Accuracy Gain**: +8-12% (from ~85% to 93-97%)

### **1.1 Optical Flow Analysis** 
**Priority**: 🔥 CRITICAL - Major missing signal
**Effort**: 2-3 days
**Impact**: Very High

#### **Technical Implementation:**
```python
def calculate_optical_flow_features(video_path, num_samples=12):
    """
    Extract temporal motion inconsistency features using OpenCV optical flow
    
    Returns:
    - flow_global_mean: Average motion magnitude across frames
    - flow_global_std: Motion consistency (low = smooth, high = jittery)
    - flow_patch_variance_mean: Local motion inconsistency
    - background_vs_foreground_ratio: Motion separation quality
    - temporal_flow_jitter_index: Frame-to-frame flow direction changes
    """
```

#### **AI Detection Signals:**
- **Background/Foreground Flow Mismatch**: AI often moves background and foreground with identical flow when they shouldn't
- **Micro-Jitter**: Flow that "jitters" at small scale while macro motion feels smooth
- **Flow Teleportation**: Sudden direction changes without physical acceleration
- **Unnatural Smoothness**: Too-perfect motion without realistic camera shake

#### **Integration Points:**
- Add to `calculate_lightweight_trajectory_metrics()` function
- Feed features to both Gemini prompt and future ensemble classifier
- Boost curvature scores when flow anomalies detected

---

### **1.2 Restructured Gemini Prompt System**
**Priority**: 🔥 CRITICAL - Fixes flip-flopping behavior  
**Effort**: 2-3 days
**Impact**: Very High

#### **Current Problems:**
- "Assume AI if text present" → guaranteed false positives
- "Default to real when ambiguous" → fights with above rule
- No structured outputs → hard to calibrate/ensemble
- Gemini makes final decision → single point of failure

#### **New Architecture:**
```python
# 1. Feed Gemini structured numeric context
numeric_context = {
    "restrav": {
        "curvature_mean": 107.1,
        "curvature_std": 23.4,
        "distance_mean": 0.41
    },
    "optical_flow": {
        "global_mean": 1.2,
        "jitter_index": 0.48,
        "bg_fg_ratio": 0.23
    },
    "metadata": {
        "has_ai_keywords": True,
        "keyword_count": 3
    }
}

# 2. Require structured output
expected_output = {
    "scores": {
        "temporal": 0-100,
        "physics": 0-100, 
        "text_anomalies": 0-100,
        "camera_artifacts": 0-100,
        "ai_style": 0-100
    },
    "evidence_for_ai": [
        {"area": "text_anomalies", "strength": "strong", "detail": "..."}
    ],
    "evidence_for_real": [
        {"area": "camera_artifacts", "strength": "medium", "detail": "..."}
    ],
    "prob_ai": 0.0-1.0,
    "prob_real": 0.0-1.0,
    "label": "ai" | "real" | "uncertain"
}
```

#### **New Prompt Rules:**
- **Multiple Evidence Required**: Single weak glitch cannot push prob_ai > 0.6
- **Balanced Assessment**: Must provide both "for AI" and "for real" evidence
- **Text Nuance**: Minor text glitches under motion blur ≠ AI classification
- **Uncertainty Handling**: If strong evidence exists for both sides → "uncertain"

---

### **1.3 Dedicated OCR Pipeline**
**Priority**: 🔥 CRITICAL - Text is strongest AI signal
**Effort**: 2-3 days  
**Impact**: Very High

#### **Technical Implementation:**
```python
def analyze_text_stability(video_path, frames_base64):
    """
    Dedicated OCR analysis separate from Gemini
    
    Uses: PaddleOCR or EasyOCR
    
    Returns:
    - ocr_char_error_rate: % non-ASCII/weird glyphs
    - ocr_frame_stability_score: Text consistency across frames
    - ocr_text_mutation_rate: How often text changes
    - ocr_unique_string_count: Number of distinct text strings
    - per_string_stability: Stability of each detected word
    """
```

#### **AI Detection Signals:**
- **Frame-to-Frame Mutations**: Real text stays identical, AI text drifts
- **Character Anomalies**: Impossible ligatures, weird glyphs
- **Kerning Drift**: Letter spacing changes across frames
- **Semantic Inconsistencies**: Real words in nonsensical combinations

#### **Integration:**
- Run OCR on same frames sent to Gemini
- Feed numeric OCR stats to Gemini as context
- Use as strong signal in ensemble classifier
- Provide detailed text analysis in results

---

### **Phase 1 Expected Outcomes:**
- ✅ **Eliminate flip-flopping** between "everything AI" and "everything real"
- ✅ **Add major missing temporal signal** (optical flow)
- ✅ **Structured, calibrated outputs** from Gemini
- ✅ **Dedicated text analysis** with numeric metrics
- ✅ **8-12% accuracy improvement** on benchmark datasets

---

# **PHASE 2: Advanced Features (Medium Impact - 3 weeks)**

## **Goal**: Add sophisticated detection signals for edge cases
**Expected Accuracy Gain**: +3-5% (from ~95% to 98%+)

### **2.1 Camera/Sensor Signature Analysis**
**Priority**: 🟡 HIGH - Catches "too perfect" AI videos
**Effort**: 3-4 days

#### **Features to Extract:**
```python
def analyze_camera_signatures(video_path):
    """
    Extract camera/sensor authenticity signals
    
    Returns:
    - temporal_noise_consistency: Real cameras have correlated noise
    - blur_direction_consistency: Motion blur patterns
    - rolling_shutter_presence_score: Phone camera artifacts
    - compression_artifact_patterns: Realistic vs synthetic compression
    - sensor_noise_spectrum: Frequency analysis of noise
    """
```

#### **AI Detection Signals:**
- **Too-Clean Areas**: AI often lacks realistic sensor noise
- **Inconsistent Noise**: Noise pattern changes unrealistically
- **Perfect Motion Blur**: AI blur often too uniform/directional
- **Missing Rolling Shutter**: Real phones show skew on fast pans

---

### **2.2 Expanded ReStraV Feature Extraction**
**Priority**: 🟡 HIGH - Better than simple thresholds
**Effort**: 2-3 days

#### **Current**: Basic curvature mean + thresholds
#### **Enhanced**: Full statistical distribution analysis

```python
def extract_enhanced_restrav_features(trajectory_metrics):
    """
    Extract comprehensive trajectory statistics
    
    Returns:
    - curvature_mean, curvature_std, curvature_max, curvature_min
    - curvature_high_tail_fraction: % frames above threshold
    - curvature_distribution_skew: Asymmetry in curvature distribution
    - distance_mean, distance_std, distance_max
    - trajectory_smoothness_index: Overall motion quality
    """
```

---

### **2.3 Face/Body Landmark Dynamics**
**Priority**: 🟡 MEDIUM - Human motion analysis
**Effort**: 4-5 days

#### **Implementation:**
```python
def analyze_human_motion_dynamics(video_path):
    """
    Use MediaPipe to extract human motion authenticity
    
    Returns:
    - blink_rate, blink_duration_consistency
    - facial_expression_jitter: Micro-expression smoothness
    - joint_angle_outlier_ratio: Impossible body positions
    - human_motion_smoothness: Natural vs synthetic movement
    """
```

#### **AI Detection Signals:**
- **Unnatural Blink Patterns**: AI often has inconsistent blink timing
- **Impossible Joint Angles**: Elbow/wrist bends that violate anatomy
- **Too-Smooth Expressions**: Lack of natural micro-expressions
- **Motion Uncanny Valley**: Movement that feels "off" but looks right

---

# **PHASE 3: Ensemble System (Game Changer - 2 weeks)**

## **Goal**: Intelligent fusion of all signals for maximum accuracy
**Expected Accuracy Gain**: +2-3% (from ~98% to 99%+)

### **3.1 Lightweight Ensemble Classifier**
**Priority**: 🔥 CRITICAL - Combines everything intelligently
**Effort**: 4-5 days

#### **Architecture:**
```python
class FrameTruthEnsemble:
    def __init__(self):
        # Lightweight classifier (XGBoost/LogisticRegression)
        self.structural_classifier = None
        
    def extract_all_features(self, video_path):
        """
        Extract all numeric features for ensemble
        
        Returns feature vector:
        - ReStraV features (8 dims)
        - Optical flow features (5 dims) 
        - OCR features (4 dims)
        - Camera signature features (6 dims)
        - Metadata features (3 dims)
        Total: ~26 dimensional feature vector
        """
        
    def predict(self, video_path, frames):
        # 1. Extract all numeric features
        features = self.extract_all_features(video_path)
        
        # 2. Get structural prediction
        p_ai_structural = self.structural_classifier.predict_proba(features)[1]
        
        # 3. Get Gemini prediction with numeric context
        gemini_result = self.analyze_with_gemini(frames, features)
        p_ai_llm = gemini_result['prob_ai']
        
        # 4. Intelligent fusion
        p_ai_final = self.fuse_predictions(p_ai_structural, p_ai_llm, features)
        
        return {
            'p_ai_final': p_ai_final,
            'p_ai_structural': p_ai_structural, 
            'p_ai_llm': p_ai_llm,
            'label': self.get_label(p_ai_final),
            'confidence': self.get_confidence(p_ai_final, features),
            'detailed_analysis': gemini_result
        }
```

#### **Fusion Strategy:**
```python
def fuse_predictions(self, p_structural, p_llm, features):
    """
    Intelligent weighted combination based on feature confidence
    
    Base weights: 60% structural, 40% LLM
    Adjustments:
    - High OCR anomalies → increase LLM weight (text analysis)
    - High flow anomalies → increase structural weight (motion analysis)
    - Conflicting signals → reduce confidence, flag for review
    """
    
    base_weight_structural = 0.6
    base_weight_llm = 0.4
    
    # Adjust weights based on feature confidence
    if features['ocr_anomaly_score'] > 0.8:
        base_weight_llm += 0.1  # Trust Gemini more for text
    
    if features['flow_anomaly_score'] > 0.8:
        base_weight_structural += 0.1  # Trust numeric features more
        
    # Normalize weights
    total_weight = base_weight_structural + base_weight_llm
    w_structural = base_weight_structural / total_weight
    w_llm = base_weight_llm / total_weight
    
    return w_structural * p_structural + w_llm * p_llm
```

---

### **3.2 Calibrated Confidence Scoring**
**Priority**: 🟡 HIGH - User trust and uncertainty handling
**Effort**: 2-3 days

#### **Confidence Factors:**
- **Signal Agreement**: Structural and LLM predictions align
- **Feature Strength**: Strong vs weak evidence
- **Historical Performance**: Model performance on similar cases
- **Uncertainty Quantification**: When to say "I don't know"

#### **Output Labels:**
```python
def get_label_and_confidence(self, p_ai_final, features):
    if p_ai_final >= 0.85:
        return "AI", "High"
    elif p_ai_final >= 0.70:
        return "AI", "Medium" 
    elif p_ai_final >= 0.55:
        return "Uncertain", "Low"
    elif p_ai_final >= 0.30:
        return "Real", "Medium"
    else:
        return "Real", "High"
```

---

## 📈 **Expected Performance Improvements**

### **Accuracy Progression:**
- **Current System**: ~85% accuracy
- **After Phase 1**: ~93-97% accuracy (+8-12%)
- **After Phase 2**: ~98%+ accuracy (+3-5%)  
- **After Phase 3**: ~99%+ accuracy (+2-3%)

### **Robustness Improvements:**
- ✅ **Eliminates flip-flopping** behavior
- ✅ **Handles edge cases** (compressed video, shaky footage)
- ✅ **Catches sophisticated AI** (Sora, Runway Gen-3)
- ✅ **Provides uncertainty estimates** when unsure
- ✅ **Explainable results** with detailed evidence

---

## 🛠 **Implementation Strategy**

### **Phase 1 (Start Immediately):**
1. **Week 1**: Optical flow analysis + OCR pipeline
2. **Week 2**: Restructured Gemini prompt system
3. **Testing**: Validate each component independently

### **Phase 2 (After Phase 1 Complete):**
1. **Week 3-4**: Camera signatures + expanded ReStraV
2. **Week 5**: Human motion analysis (if needed)
3. **Testing**: Integration testing with Phase 1

### **Phase 3 (Final Integration):**
1. **Week 6**: Ensemble classifier training
2. **Week 7**: Calibration and confidence scoring
3. **Testing**: End-to-end system validation

---

## 🧪 **Testing & Validation Strategy**

### **Datasets:**
- **Benchmark Datasets**: FaceForensics++, DFDC, CelebDF
- **Modern AI Samples**: Sora, Runway Gen-3, Pika outputs
- **Real Video Corpus**: Phone footage, professional video, compressed content

### **Metrics:**
- **Accuracy**: Overall classification performance
- **Precision/Recall**: AI detection vs false positive rate
- **Calibration**: Confidence scores match actual accuracy
- **Robustness**: Performance on edge cases

### **A/B Testing:**
- **Current vs Enhanced**: Direct comparison on same dataset
- **Component Ablation**: Impact of each individual feature
- **User Studies**: Real-world usage patterns

---

## 🎯 **Success Criteria**

### **Phase 1 Success:**
- [ ] Optical flow features extracted and integrated
- [ ] Gemini outputs structured and consistent
- [ ] OCR pipeline provides numeric text metrics
- [ ] No more flip-flopping between AI/Real classifications
- [ ] 8%+ accuracy improvement on benchmark dataset

### **Phase 2 Success:**
- [ ] Camera signature analysis detects "too perfect" videos
- [ ] Enhanced ReStraV features improve edge case handling
- [ ] Human motion analysis catches subtle AI tells
- [ ] 3%+ additional accuracy improvement

### **Phase 3 Success:**
- [ ] Ensemble system combines all signals intelligently
- [ ] Calibrated confidence scores match actual performance
- [ ] System achieves 99%+ accuracy on benchmark datasets
- [ ] Robust performance on modern AI generators (Sora, etc.)

---

## 🔄 **Maintenance & Updates**

### **Continuous Improvement:**
- **Model Retraining**: Monthly updates with new AI samples
- **Feature Engineering**: Add new signals as AI evolves
- **Prompt Optimization**: Refine Gemini instructions based on performance
- **Ensemble Tuning**: Adjust fusion weights based on real-world data

### **Monitoring:**
- **Performance Tracking**: Accuracy metrics over time
- **Error Analysis**: Common failure modes and patterns
- **User Feedback**: Real-world usage insights
- **Adversarial Testing**: Robustness against sophisticated attacks

---

This roadmap provides a systematic path to transform FrameTruth into a world-class AI detection system that can compete with the best research-grade detectors while maintaining practical usability.
