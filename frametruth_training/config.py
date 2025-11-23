"""
FrameTruth Training Pipeline Configuration
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
(DATA_DIR / "raw_videos").mkdir(exist_ok=True)
(DATA_DIR / "frames").mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# API Configuration
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "135d1d8e94msh773cfcb7bd35969p1fada7jsn4743820f5475")
RAPIDAPI_HOST = "social-download-all-in-one.p.rapidapi.com"
RAPIDAPI_ENDPOINT = f"https://{RAPIDAPI_HOST}/v1/social/autolink"

# Feature extraction settings
FRAMES_PER_VIDEO = 16  # Number of frames to extract per video
FRAME_SIZE = (256, 256)  # Resize frames to this size for processing

# Training settings
TEST_SIZE = 0.2  # 20% for validation
RANDOM_STATE = 42
N_ESTIMATORS = 100  # For ensemble models

# Feature categories
FEATURE_CATEGORIES = {
    'prnu': ['prnu_mean_corr', 'prnu_std_corr', 'prnu_positive_ratio', 'prnu_consistency_score'],
    'trajectory': ['trajectory_curvature_mean', 'trajectory_curvature_std', 'trajectory_max_curvature', 'trajectory_mean_distance'],
    'optical_flow': ['flow_jitter_index', 'flow_bg_fg_ratio', 'flow_patch_variance', 'flow_smoothness_score', 'flow_global_mean', 'flow_global_std'],
    'ocr': ['ocr_char_error_rate', 'ocr_frame_stability', 'ocr_mutation_rate', 'ocr_unique_string_count', 'ocr_total_detections'],
    'frequency': ['freq_low_power_mean', 'freq_mid_power_mean', 'freq_high_power_mean', 'freq_high_low_ratio_mean', 'freq_spectrum_slope_mean'],
    'noise': ['noise_variance_r_mean', 'noise_variance_g_mean', 'noise_variance_b_mean', 'cross_channel_corr_rg_mean', 'spatial_autocorr_mean', 'temporal_noise_consistency'],
    'codec': ['avg_bitrate', 'i_frame_ratio', 'p_frame_ratio', 'b_frame_ratio', 'gop_length_mean', 'gop_length_std', 'double_compression_score'],
    'metadata': ['duration_seconds', 'fps', 'resolution_width', 'resolution_height']
}

# Model thresholds
AI_THRESHOLD = 0.8  # Above this = AI
REAL_THRESHOLD = 0.2  # Below this = Real
# Between thresholds = Uncertain

print(f"FrameTruth Training Pipeline initialized")
print(f"Data directory: {DATA_DIR}")
print(f"Models directory: {MODELS_DIR}")
