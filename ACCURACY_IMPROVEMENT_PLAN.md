# 🎯 FRAMETRUTH ACCURACY IMPROVEMENT PLAN
## Building the Most Accurate AI Video Detector Ever

### 🚨 CURRENT PROBLEM ANALYSIS

**Root Issue**: Uncalibrated heuristics stacked on top of Gemini are canceling out benefits
- Hand-picked thresholds (0.15, 1.5, 0.8) are arbitrary guesses
- Big point weights (+40, +35, +30) are not data-driven
- Features are valuable but implementation is counterproductive

### 📊 NEW ARCHITECTURE: DATA-DRIVEN APPROACH

#### Phase 1: Feature Extraction Pipeline (Keep & Improve)
```python
# Keep computing these features - they're scientifically sound
features = {
    'trajectory': {
        'curvature_mean': float,
        'curvature_std': float, 
        'max_curvature': float
    },
    'optical_flow': {
        'jitter_index': float,
        'bg_fg_ratio': float,
        'patch_variance': float,
        'smoothness_score': float
    },
    'ocr_text': {
        'char_error_rate': float,
        'frame_stability': float,
        'mutation_rate': float,
        'unique_string_count': int
    },
    'prnu_sensor': {
        'mean_correlation': float,
        'std_correlation': float,
        'positive_ratio': float,
        'consistency_score': float
    }
}
```

#### Phase 2: Calibration Dataset Collection
```
Target: 200-500 videos total
├── Real Videos (100-250)
│   ├── Phone recordings (iPhone, Android)
│   ├── Professional cameras (DSLR, mirrorless)
│   ├── Webcams & security cameras
│   ├── Action cameras (GoPro, DJI)
│   └── Various conditions (indoor, outdoor, low light)
└── AI Videos (100-250)
    ├── Sora generations
    ├── Runway Gen-3
    ├── Pika Labs
    ├── Kling AI
    ├── Luma Dream Machine
    └── Other generators (Haiper, Veo, etc.)
```

#### Phase 3: New Scoring Logic (No More Arbitrary Points)
```python
def evaluate_video(features, gemini_prob):
    # Step 1: Compute structural flags (calibrated thresholds)
    flags_ai = count_ai_flags(features)
    flags_real = count_real_flags(features)
    
    # Step 2: Fusion logic (not addition)
    if gemini_prob >= 0.85:  # Gemini confident AI
        if flags_real >= 2 and flags_ai == 0:
            return "AI (but structural signals suggest real-like patterns)"
        else:
            return "AI"
    
    elif gemini_prob <= 0.15:  # Gemini confident Real
        if flags_ai >= 2:
            return "Uncertain (Gemini says real, structure says AI)"
        else:
            return "Real"
    
    else:  # Gemini uncertain (0.15 < prob < 0.85)
        if flags_ai > flags_real + 1:
            return "Likely AI"
        elif flags_real > flags_ai + 1:
            return "Likely Real"
        else:
            return "Uncertain"
```

### 🛠️ IMPLEMENTATION ROADMAP

#### Week 1: Data Collection & Calibration
1. **Build calibration dataset** (200+ videos)
2. **Extract features** from all videos
3. **Analyze distributions** (real vs AI)
4. **Set evidence-based thresholds**

#### Week 2: New Scoring System
1. **Remove arbitrary point system**
2. **Implement flag-based logic**
3. **Add uncertainty handling**
4. **Calibrate fusion weights**

#### Week 3: Advanced ML Integration
1. **Train lightweight classifier** on structural features
2. **Implement ensemble approach** (Gemini + Structural)
3. **Add confidence calibration**
4. **Validate on holdout set**

### 📋 SPECIFIC FIXES NEEDED

#### 1. Metadata Analysis - DIAL WAY DOWN
```python
# OLD: +35-40 points for AI keywords
# NEW: Tiny nudge only for explicit generators
def check_metadata(title, description, tags):
    explicit_generators = ['sora', 'runway gen-3', 'pika labs', 'generated with']
    if any(gen in text.lower() for gen in explicit_generators 
           for text in [title, description, ' '.join(tags)]):
        return 0.05  # 5% nudge, not 40 points
    return 0.0
```

#### 2. Trajectory Analysis - Keep Features, Fix Usage
```python
# OLD: Arbitrary buckets (>130° → +40, >110° → +30)
# NEW: Continuous features with calibrated thresholds
def trajectory_flags(curvature_mean, curvature_std):
    flags = 0
    # These thresholds will be set from real data analysis
    if curvature_mean > CALIBRATED_THRESHOLD_HIGH:  # e.g., 95th percentile of real videos
        flags += 1
    if curvature_std > CALIBRATED_THRESHOLD_STD:
        flags += 1
    return flags
```

#### 3. OCR Analysis - Keep High Value, Refine Usage
```python
# OLD: Single anomaly overrides everything
# NEW: Severity-based evidence
def ocr_flags(char_error_rate, frame_stability, mutation_rate):
    flags = 0
    # Severe anomalies (multiple indicators)
    if char_error_rate > 0.15 and frame_stability < 0.6:
        flags += 2  # Strong evidence
    elif char_error_rate > 0.1 or frame_stability < 0.7:
        flags += 1  # Moderate evidence
    
    if mutation_rate > 0.4:  # Very high mutation
        flags += 1
    
    return min(flags, 3)  # Cap at 3 flags max
```

#### 4. PRNU Analysis - Fix Overconfidence
```python
# OLD: "Decisive evidence of authenticity"
# NEW: "Supports hypothesis, not proof"
def prnu_flags(mean_corr, std_corr, positive_ratio):
    flags_ai = 0
    flags_real = 0
    
    # Evidence for AI (no stable sensor pattern)
    if mean_corr < 0.05 or positive_ratio < 0.5:
        flags_ai += 1
    
    # Evidence for real (stable pattern, but not proof)
    elif mean_corr > CALIBRATED_REAL_THRESHOLD and std_corr < 0.08:
        flags_real += 1
    
    return flags_ai, flags_real
```

### 🎯 SUCCESS METRICS

#### Accuracy Targets
- **Overall Accuracy**: >95% on balanced test set
- **False Positive Rate**: <3% (real videos called AI)
- **False Negative Rate**: <5% (AI videos called real)
- **Uncertainty Rate**: 10-15% (better to be uncertain than wrong)

#### Robustness Tests
- **Cross-generator**: Train on Sora, test on Runway
- **Cross-quality**: Train on HD, test on compressed
- **Adversarial**: Test on videos designed to fool detectors

### 🚀 IMMEDIATE NEXT STEPS

1. **Create calibration dataset collection script**
2. **Build feature extraction pipeline**
3. **Implement new scoring logic**
4. **Remove arbitrary point system**
5. **Add proper uncertainty handling**

This approach will give us:
- ✅ Evidence-based thresholds
- ✅ Proper uncertainty quantification  
- ✅ Robust fusion of multiple signals
- ✅ Calibrated confidence scores
- ✅ The most accurate detector possible

Ready to implement this systematic approach?
