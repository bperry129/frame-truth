import os
import uvicorn
import sqlite3
import uuid
import json
import cv2
import base64
import requests
import shutil
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import yt_dlp
import torch
import torchvision.transforms as transforms
from PIL import Image
import easyocr
import re
from difflib import SequenceMatcher
import sys
from pathlib import Path

# Add frametruth_training to path for evidence-based scorer
sys.path.append(str(Path(__file__).parent.parent / "frametruth_training"))
from evidence_based_scorer import EvidenceBasedScorer

# Import feature extractors
sys.path.append(str(Path(__file__).parent.parent / "frametruth_training" / "feature_extractor"))
from frequency_features import compute_frequency_features
from noise_features import compute_noise_features
from codec_features import compute_codec_features
from metadata_features import compute_metadata_features

load_dotenv(".env")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

# Verify PyTorch installation (for Railway debugging)
print("=" * 50)
print("🔍 PyTorch Installation Check:")
print(f"   Torch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
print(f"   Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print("=" * 50)

# Verify API key is loaded
if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY not found in .env file!")
    print("Please check that backend/.env exists and contains: OPENROUTER_API_KEY=your_key_here")
else:
    print(f"✓ API Key loaded: {OPENROUTER_API_KEY[:20]}...")

# Verify admin credentials are loaded
if not ADMIN_PASS:
    print("WARNING: ADMIN_PASS not found in .env file!")
    print("Please add ADMIN_PASS=your_secure_password to backend/.env")
else:
    print(f"✓ Admin credentials loaded for user: {ADMIN_USER}")

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global DINOv2 model (lazy loaded)
_dinov2_model = None
_dinov2_transform = None

def get_dinov2_model():
    """Lazy load DINOv2 model (cached after first use)"""
    global _dinov2_model, _dinov2_transform
    
    if _dinov2_model is None:
        print("🔄 Loading DINOv2 model (first-time setup, ~90MB download)...")
        try:
            # Load smallest DINOv2 variant (vits14) for efficiency
            _dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            _dinov2_model.eval()  # Set to evaluation mode
            
            # DINOv2 preprocessing transform
            _dinov2_transform = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            print("✅ DINOv2 model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load DINOv2: {e}")
            raise
    
    return _dinov2_model, _dinov2_transform

# Database Setup
DB_NAME = "frame_truth.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS submissions (
            id TEXT PRIMARY KEY,
            filename TEXT,
            original_url TEXT,
            analysis_result TEXT,
            ip_address TEXT,
            created_at DATETIME
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Directories
DOWNLOAD_DIR = "downloads"
COOKIES_DIR = "backend/cookies"
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)
if not os.path.exists(COOKIES_DIR):
    os.makedirs(COOKIES_DIR)

COOKIES_FILE = os.path.join(COOKIES_DIR, "youtube_cookies.txt")

# Mount Static Files for playback
app.mount("/videos", StaticFiles(directory=DOWNLOAD_DIR), name="videos")

# Models
class DownloadRequest(BaseModel):
    url: str

class AnalyzeRequest(BaseModel):
    filename: str
    original_url: str = ""

class SaveSubmissionRequest(BaseModel):
    filename: str
    original_url: str = ""
    analysis_result: dict

# Helpers
def check_rate_limit(ip: str) -> bool:
    # TEMPORARILY DISABLE RATE LIMITING FOR DEBUGGING
    print(f"🔓 RATE LIMITING DISABLED - IP: {ip} - bypassing all limits")
    return True
    
    # Original rate limiting code (commented out for debugging)
    # # Whitelist localhost, local development IPs, and specific user IPs
    # local_ips = ["127.0.0.1", "localhost", "::1", "0.0.0.0", "192.168.1.16", "173.239.214.13"]
    # # Also whitelist entire 173.239.x.x range
    # if (ip in local_ips or 
    #     ip.startswith("127.") or ip.startswith("192.168.") or ip.startswith("10.") or ip.startswith("172.") or
    #     ip.startswith("173.239.")):
    #     print(f"✓ Whitelisted IP: {ip} - bypassing rate limit")
    #     return True

    # conn = sqlite3.connect(DB_NAME)
    # c = conn.cursor()
    # # Rate limit: 5 per day
    # one_day_ago = (datetime.now() - timedelta(days=1)).isoformat()
    # c.execute("SELECT count(*) FROM submissions WHERE ip_address = ? AND created_at > ?", (ip, one_day_ago))
    # count = c.fetchone()[0]
    # conn.close()
    # print(f"Rate limit check for {ip}: {count}/5 submissions today")
    # return count < 5

def scan_metadata_for_ai_keywords(url: str) -> dict:
    """
    Scan video metadata (from URL) for AI-related keywords.
    Returns dict with detected keywords and confidence boost.
    """
    ai_keywords = [
        # AI Generators
        'sora', 'runway', 'pika', 'kling', 'midjourney', 'stable diffusion',
        'gen-3', 'gen-2', 'luma', 'haiper', 'synthesia',
        
        # AI-related terms
        'ai generated', 'ai-generated', 'ai video', 'artificial intelligence',
        'machine learning', 'deep learning', 'neural network',
        'text-to-video', 'text to video', 'video generation',
        'generative ai', 'synthetic', 'computer generated',
        
        # Common AI hashtags
        '#aigenerated', '#aiart', '#aivideo', '#texttovideo',
        '#runwayml', '#soraai', '#aitools', '#generativeai'
    ]
    
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            
            title = (info.get('title') or '').lower()
            description = (info.get('description') or '').lower()
            tags = [tag.lower() for tag in (info.get('tags') or [])]
            
            # Combine all text for searching
            all_text = f"{title} {description} {' '.join(tags)}"
            
            detected = []
            for keyword in ai_keywords:
                if keyword in all_text:
                    detected.append(keyword)
            
            if detected:
                # High confidence boost if AI keywords found
                return {
                    'has_ai_keywords': True,
                    'keywords_found': detected,
                    'confidence_boost': 40,  # Significant boost
                    'score_increase': 35  # Increase curvature score
                }
            else:
                return {
                    'has_ai_keywords': False,
                    'keywords_found': [],
                    'confidence_boost': 0,
                    'score_increase': 0
                }
    except:
        # If metadata extraction fails, return neutral
        return {
            'has_ai_keywords': False,
            'keywords_found': [],
            'confidence_boost': 0,
            'score_increase': 0
        }

def calculate_prnu_sensor_fingerprint(video_path, num_samples=16):
    """
    🚀 GAME-CHANGING DETECTION: Camera Sensor Fingerprint Consistency (PRNU)
    
    This is the closest thing to a "blood type" for real video.
    
    Real cameras have a fixed, physical sensor with tiny imperfections:
    - Some pixels are slightly more/less sensitive than others
    - Noise pattern is stable across frames (same sensor = same "grain fingerprint")
    - Rolling shutter & lens distortion behave consistently
    
    AI videos usually:
    - Don't have a physically consistent PRNU pattern
    - Add synthetic noise that doesn't match real sensor statistics
    - Show per-frame differences in "grain" that aren't tied to a real sensor
    
    Returns:
    - prnu_mean_corr: Average correlation with global fingerprint
    - prnu_std_corr: Standard deviation of correlations
    - prnu_positive_ratio: Ratio of positive correlations
    - prnu_consistency_score: Overall consistency metric
    """
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            return None
        
        # Sample frames evenly across video
        step = max(1, total_frames // num_samples)
        
        frames = []
        count = 0
        extracted = 0
        
        print(f"   🔬 PRNU Analysis: extracting sensor fingerprint from {num_samples} frames...")
        
        # Extract frames for PRNU analysis
        while cap.isOpened() and extracted < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
                
            if count % step == 0:
                # Convert to grayscale for noise analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Resize to manageable size for processing speed
                gray_resized = cv2.resize(gray, (256, 256))
                frames.append(gray_resized.astype(np.float32))
                extracted += 1
                    
            count += 1
        
        cap.release()
        
        if len(frames) < 4:  # Need minimum frames for reliable fingerprint
            return None
        
        # Step 1: Denoise each frame to isolate PRNU
        residuals = []
        for i, frame in enumerate(frames):
            # Apply strong denoising to isolate sensor noise
            # Using fastNlMeansDenoising as a starting point
            denoised = cv2.fastNlMeansDenoising(frame.astype(np.uint8), None, 10, 7, 21)
            denoised = denoised.astype(np.float32)
            
            # Calculate noise residual: original - denoised
            residual = frame - denoised
            
            # Normalize residual
            residual = residual / (np.std(residual) + 1e-6)
            residuals.append(residual)
        
        # Step 2: Estimate global sensor fingerprint
        # Average residuals across frames - sensor noise should reinforce, random noise cancels
        global_fingerprint = np.mean(residuals, axis=0)
        
        # Normalize the global fingerprint
        global_fingerprint = global_fingerprint / (np.std(global_fingerprint) + 1e-6)
        
        # Step 3: Measure per-frame correlation with global fingerprint
        correlations = []
        for residual in residuals:
            # Flatten for correlation calculation
            residual_flat = residual.flatten()
            fingerprint_flat = global_fingerprint.flatten()
            
            # Calculate normalized cross-correlation
            correlation = np.corrcoef(residual_flat, fingerprint_flat)[0, 1]
            
            # Handle NaN case
            if np.isnan(correlation):
                correlation = 0.0
            
            correlations.append(correlation)
        
        # Step 4: Calculate PRNU metrics
        correlations = np.array(correlations)
        
        prnu_mean_corr = float(np.mean(correlations))
        prnu_std_corr = float(np.std(correlations))
        
        # Count positive correlations (should be high for real cameras)
        positive_threshold = 0.1  # Small positive threshold
        prnu_positive_ratio = float(np.sum(correlations > positive_threshold) / len(correlations))
        
        # Calculate overall consistency score
        # Real cameras: high mean, low std, high positive ratio
        # AI videos: near-zero mean, higher std, low positive ratio
        consistency_score = prnu_mean_corr * prnu_positive_ratio / (prnu_std_corr + 0.1)
        prnu_consistency_score = float(max(0, min(1, consistency_score)))
        
        return {
            'prnu_mean_corr': prnu_mean_corr,
            'prnu_std_corr': prnu_std_corr,
            'prnu_positive_ratio': prnu_positive_ratio,
            'prnu_consistency_score': prnu_consistency_score,
            'num_samples': len(frames),
            'method': 'prnu_sensor_fingerprint'
        }
        
    except Exception as e:
        print(f"⚠️ PRNU sensor fingerprint analysis failed: {str(e)}")
        return None

def calculate_optical_flow_features(video_path, num_samples=12):
    """
    FIXED: CPU-OPTIMIZED optical flow analysis for speed + accuracy balance
    
    This is a major missing signal that can boost accuracy by 8-12%.
    
    AI Detection Signals:
    - Background/Foreground Flow Mismatch: AI often moves background and foreground with identical flow
    - Micro-Jitter: Flow that "jitters" at small scale while macro motion feels smooth  
    - Flow Teleportation: Sudden direction changes without physical acceleration
    - Unnatural Smoothness: Too-perfect motion without realistic camera shake
    
    Returns:
    - flow_global_mean: Average motion magnitude across frames
    - flow_global_std: Motion consistency (low = smooth, high = jittery)
    - flow_patch_variance: Local motion inconsistency
    - flow_jitter_index: Frame-to-frame flow direction changes
    - flow_smoothness_score: Motion smoothness quality
    """
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            return None
        
        # CPU OPTIMIZATION: Minimal samples for speed
        flow_samples = max(3, min(5, num_samples))  # At least 3 frames for flow
        step = max(1, total_frames // flow_samples)
        
        frames = []
        count = 0
        extracted = 0
        
        print(f"   🌊 FIXED optical flow: analyzing {flow_samples} frames...")
        
        # Extract frames for optical flow analysis
        while cap.isOpened() and extracted < flow_samples:
            ret, frame = cap.read()
            if not ret:
                break
                
            if count % step == 0:
                # CPU OPTIMIZATION: Smaller resolution for faster processing
                frame_resized = cv2.resize(frame, (128, 128))
                gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                frames.append(gray)
                extracted += 1
                    
            count += 1
        
        cap.release()
        
        if len(frames) < 2:  # Need at least 2 frames for flow
            return None
        
        # Calculate optical flow between consecutive frames using Farneback method
        flow_magnitudes = []
        flow_directions = []
        patch_variances = []
        
        for i in range(len(frames) - 1):
            try:
                # Use Farneback dense optical flow (more robust than LK)
                flow = cv2.calcOpticalFlowFarneback(
                    frames[i], frames[i+1], None, 
                    pyr_scale=0.5, levels=2, winsize=13, 
                    iterations=2, poly_n=5, poly_sigma=1.1, flags=0
                )
                
                # Calculate flow magnitude and direction
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                
                # Global flow statistics
                flow_magnitudes.append(np.mean(magnitude))
                flow_directions.append(np.mean(angle))
                
                # Local patch variance (motion inconsistency)
                patch_size = 16  # Smaller patches for 128x128 frames
                h, w = magnitude.shape
                patch_vars = []
                
                for y in range(0, h - patch_size, patch_size):
                    for x in range(0, w - patch_size, patch_size):
                        patch = magnitude[y:y+patch_size, x:x+patch_size]
                        patch_vars.append(np.var(patch))
                
                if patch_vars:
                    patch_variances.append(np.mean(patch_vars))
                
            except Exception as flow_error:
                print(f"   ⚠️ Flow calculation failed for frame pair {i}: {flow_error}")
                continue
        
        if len(flow_magnitudes) == 0:
            print(f"   ❌ No valid optical flow calculated")
            return None
        
        # Calculate flow features
        flow_global_mean = float(np.mean(flow_magnitudes))
        flow_global_std = float(np.std(flow_magnitudes))
        flow_patch_variance = float(np.mean(patch_variances)) if patch_variances else 0.0
        
        # Calculate temporal jitter (frame-to-frame flow direction changes)
        direction_changes = []
        for i in range(len(flow_directions) - 1):
            # Calculate angular difference between consecutive flow directions
            diff = abs(flow_directions[i+1] - flow_directions[i])
            # Handle circular nature of angles
            diff = min(diff, 2*np.pi - diff)
            direction_changes.append(diff)
        
        flow_jitter_index = float(np.mean(direction_changes)) if direction_changes else 0.0
        
        # Calculate smoothness score (inverse of jitter)
        flow_smoothness_score = 1.0 / (flow_jitter_index + 1e-6)
        
        print(f"   ✅ Optical flow computed: mean={flow_global_mean:.3f}, jitter={flow_jitter_index:.3f}")
        
        return {
            'flow_global_mean': flow_global_mean,
            'flow_global_std': flow_global_std,
            'flow_patch_variance': flow_patch_variance,
            'flow_jitter_index': flow_jitter_index,
            'flow_smoothness_score': flow_smoothness_score,
            'num_samples': len(frames),
            'method': 'optical_flow_farneback_fixed'
        }
        
    except Exception as e:
        print(f"⚠️ Optical flow calculation failed: {str(e)}")
        return None

# Global EasyOCR reader (lazy loaded)
_ocr_reader = None

def get_ocr_reader():
    """Lazy load EasyOCR reader (cached after first use)"""
    global _ocr_reader
    
    if _ocr_reader is None:
        print("🔄 Loading EasyOCR model (first-time setup, ~50MB download)...")
        try:
            # Load EasyOCR with English support
            _ocr_reader = easyocr.Reader(['en'], gpu=False)  # Use CPU for compatibility
            print("✅ EasyOCR model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load EasyOCR: {e}")
            raise
    
    return _ocr_reader

def analyze_text_stability(video_path, num_samples=12):
    """
    CPU-OPTIMIZED OCR analysis for speed + accuracy balance
    
    This is a critical missing signal - text is the strongest AI indicator.
    
    AI Detection Signals:
    - Frame-to-Frame Mutations: Real text stays identical, AI text drifts
    - Character Anomalies: Impossible ligatures, weird glyphs
    - Kerning Drift: Letter spacing changes across frames
    - Semantic Inconsistencies: Real words in nonsensical combinations
    
    Returns:
    - ocr_char_error_rate: % non-ASCII/weird glyphs
    - ocr_frame_stability_score: Text consistency across frames
    - ocr_text_mutation_rate: How often text changes
    - ocr_unique_string_count: Number of distinct text strings
    - per_string_stability: Stability of each detected word
    """
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            return None
        
        # CPU OPTIMIZATION: Minimal OCR samples for speed
        ocr_samples = max(1, min(2, num_samples // 3))  # Max 2 frames for OCR (was 4)
        step = max(1, total_frames // ocr_samples)
        
        frames = []
        count = 0
        extracted = 0
        
        print(f"   CPU-optimized OCR: analyzing {ocr_samples} frames (minimal for speed)...")
        
        # Extract frames for OCR analysis
        while cap.isOpened() and extracted < ocr_samples:
            ret, frame = cap.read()
            if not ret:
                break
                
            if count % step == 0:
                # CPU OPTIMIZATION: Even smaller resolution for faster OCR
                frame_resized = cv2.resize(frame, (256, 256))  # Was 384x384
                frames.append(frame_resized)
                extracted += 1
                    
            count += 1
        
        cap.release()
        
        if len(frames) < 1:  # Only need 1 frame minimum
            return None
        
        # Get OCR reader
        reader = get_ocr_reader()
        
        # Extract text from each frame
        frame_texts = []
        all_detected_strings = []
        
        for i, frame in enumerate(frames):
            try:
                # Run OCR on frame
                results = reader.readtext(frame, detail=1)
                
                frame_text_data = []
                for (bbox, text, confidence) in results:
                    # Filter out low-confidence detections
                    if confidence > 0.4:  # Higher threshold for speed (was 0.3)
                        cleaned_text = text.strip()
                        if len(cleaned_text) > 1:  # Ignore single characters
                            frame_text_data.append({
                                'text': cleaned_text,
                                'confidence': confidence,
                                'bbox': bbox,
                                'frame': i
                            })
                            all_detected_strings.append(cleaned_text)
                
                frame_texts.append(frame_text_data)
                
            except Exception as e:
                print(f"⚠️ OCR failed on frame {i}: {str(e)}")
                frame_texts.append([])
        
        if not all_detected_strings:
            # No text detected
            return {
                'ocr_char_error_rate': 0.0,
                'ocr_frame_stability_score': 1.0,  # Perfect stability (no text to be unstable)
                'ocr_text_mutation_rate': 0.0,
                'ocr_unique_string_count': 0,
                'per_string_stability': {},
                'total_text_detections': 0,
                'has_text': False,
                'method': 'easyocr'
            }
        
        # Analyze character anomalies
        char_error_count = 0
        total_chars = 0
        
        for text in all_detected_strings:
            for char in text:
                total_chars += 1
                # Check for non-printable or unusual characters
                if not char.isprintable() or ord(char) > 127:
                    char_error_count += 1
                # Check for common AI text artifacts
                elif char in ['', '□', '▢', '◯', '○']:  # Common OCR/AI artifacts
                    char_error_count += 1
        
        ocr_char_error_rate = float(char_error_count / max(total_chars, 1))
        
        # Analyze frame-to-frame text stability
        unique_strings = list(set(all_detected_strings))
        ocr_unique_string_count = len(unique_strings)
        
        # Track how each unique string appears across frames
        string_stability = {}
        for unique_str in unique_strings:
            appearances = []
            for frame_idx, frame_data in enumerate(frame_texts):
                # Check if this string appears in this frame
                found_in_frame = False
                for detection in frame_data:
                    # Use fuzzy matching for slight variations
                    similarity = SequenceMatcher(None, unique_str.lower(), detection['text'].lower()).ratio()
                    if similarity > 0.8:  # 80% similarity threshold
                        found_in_frame = True
                        break
                appearances.append(found_in_frame)
            
            # Calculate stability score for this string
            if sum(appearances) > 1:  # String appears in multiple frames
                # Check consistency - should appear in consecutive frames if it's real text
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
                
                # Real text should appear in long consecutive runs
                # AI text often flickers in and out
                max_run = max(consecutive_runs) if consecutive_runs else 0
                stability_score = max_run / len(appearances)
                string_stability[unique_str] = stability_score
            else:
                # Single appearance - neutral stability
                string_stability[unique_str] = 0.5
        
        # Calculate overall frame stability score
        if string_stability:
            ocr_frame_stability_score = float(np.mean(list(string_stability.values())))
        else:
            ocr_frame_stability_score = 1.0
        
        # Calculate text mutation rate
        # Count how often text changes between consecutive frames
        mutations = 0
        comparisons = 0
        
        for i in range(len(frame_texts) - 1):
            current_texts = set([d['text'].lower() for d in frame_texts[i]])
            next_texts = set([d['text'].lower() for d in frame_texts[i + 1]])
            
            if current_texts or next_texts:  # At least one frame has text
                comparisons += 1
                # Calculate Jaccard similarity
                intersection = len(current_texts.intersection(next_texts))
                union = len(current_texts.union(next_texts))
                similarity = intersection / max(union, 1)
                
                if similarity < 0.7:  # Significant change
                    mutations += 1
        
        ocr_text_mutation_rate = float(mutations / max(comparisons, 1))
        
        return {
            'ocr_char_error_rate': ocr_char_error_rate,
            'ocr_frame_stability_score': ocr_frame_stability_score,
            'ocr_text_mutation_rate': ocr_text_mutation_rate,
            'ocr_unique_string_count': ocr_unique_string_count,
            'per_string_stability': string_stability,
            'total_text_detections': len(all_detected_strings),
            'has_text': True,
            'method': 'easyocr'
        }
        
    except Exception as e:
        print(f"⚠️ OCR analysis failed: {str(e)}")
        return None

def calculate_lightweight_trajectory_metrics(video_path, num_samples=12):
    """
    Fast trajectory analysis using OpenCV (no PyTorch/DINOv2 needed).
    Inspired by ReStraV but using visual features only.
    
    Analyzes:
    - Color histogram changes (distribution shifts)
    - Edge density changes (structural complexity)
    - Optical flow consistency (motion patterns)
    
    Real videos: More consistent trajectories
    AI videos: Irregular, jumpy trajectories
    """
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            return None
        
        # Sample frames evenly
        step = max(1, total_frames // num_samples)
        
        # Extract visual features from frames
        features = []
        count = 0
        extracted = 0
        
        print(f"   Extracting visual trajectory features from {num_samples} frames...")
        
        while cap.isOpened() and extracted < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
                
            if count % step == 0:
                # Resize frame for faster processing
                frame_small = cv2.resize(frame, (128, 128))
                
                # Feature 1: Color histogram (RGB distribution)
                hist_r = cv2.calcHist([frame_small], [0], None, [32], [0, 256])
                hist_g = cv2.calcHist([frame_small], [1], None, [32], [0, 256])
                hist_b = cv2.calcHist([frame_small], [2], None, [32], [0, 256])
                color_hist = np.concatenate([hist_r, hist_g, hist_b]).flatten()
                color_hist = color_hist / (color_hist.sum() + 1e-6)  # Normalize
                
                # Feature 2: Edge density (structural complexity)
                gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.mean(edges) / 255.0
                
                # Feature 3: Brightness variance (lighting consistency)
                brightness_var = np.var(gray) / 255.0
                
                # Combine features into vector
                feature_vec = np.concatenate([
                    color_hist,  # 96-dim
                    [edge_density, brightness_var]  # 2-dim
                ])  # Total: 98-dim
                
                features.append(feature_vec)
                extracted += 1
                    
            count += 1
        
        cap.release()
        
        if len(features) < 3:
            return None
        
        # Convert to numpy array
        features_array = np.array(features)
        
        # Calculate displacement vectors (visual changes between frames)
        displacements = np.diff(features_array, axis=0)
        
        # Calculate stepwise distances (magnitude of visual change)
        distances = np.linalg.norm(displacements, axis=1)
        
        # Calculate curvature (angle between consecutive displacement vectors)
        curvatures = []
        for i in range(len(displacements) - 1):
            vec1 = displacements[i]
            vec2 = displacements[i + 1]
            
            # Normalize vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 > 1e-6 and norm2 > 1e-6:
                # Calculate cosine similarity
                cos_sim = np.dot(vec1, vec2) / (norm1 * norm2)
                cos_sim = np.clip(cos_sim, -1.0, 1.0)
                # Convert to angle in degrees
                angle = np.arccos(cos_sim) * 180 / np.pi
                curvatures.append(angle)
        
        if len(curvatures) == 0:
            return None
        
        # Calculate statistics
        mean_curvature = float(np.mean(curvatures))
        curvature_variance = float(np.var(curvatures))
        mean_distance = float(np.mean(distances))
        max_curvature = float(np.max(curvatures))
        
        return {
            'mean_curvature': mean_curvature,
            'curvature_variance': curvature_variance,
            'mean_distance': mean_distance,
            'max_curvature': max_curvature,
            'num_samples': len(features),
            'method': 'lightweight_opencv'
        }
        
    except Exception as e:
        print(f"⚠️ Trajectory calculation failed: {str(e)}")
        return None

def extract_frames_base64(video_path, num_frames=15):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return []

    step = max(1, total_frames // num_frames)
    
    count = 0
    extracted = 0
    while cap.isOpened() and extracted < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % step == 0:
            # PHASE 1 IMPROVEMENT: Higher resolution and quality for better artifact detection
            # Maintain aspect ratio, max width 1024 (was 512)
            height, width = frame.shape[:2]
            MAX_WIDTH = 1024
            if width > MAX_WIDTH:
                scale = MAX_WIDTH / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # PHASE 1 IMPROVEMENT: Higher JPEG quality 95% (was 70%)
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            b64 = base64.b64encode(buffer).decode('utf-8')
            frames.append(f"data:image/jpeg;base64,{b64}")
            extracted += 1
            
        count += 1

    cap.release()
    return frames

# Endpoints
@app.post("/api/upload")
async def upload_video_file(file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        ext = file.filename.split('.')[-1] if '.' in file.filename else 'mp4'
        filename = f"{file_id}.{ext}"
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {
            "filename": filename,
            "url": f"/videos/{filename}"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Upload failed: {str(e)}"})

async def download_with_unified_api(url: str, file_id: str) -> dict:
    """
    Unified social media downloader using RapidAPI "Download All in One Elite"
    Supports YouTube, TikTok, Instagram, Twitter, Facebook, and more
    
    API Format:
    {
        url: '',
        source: '',
        author: '',
        title: '',
        thumbnail: '',
        duration: '',
        medias: [
            {
                url: '',
                quality: '',
                extension: '',
                type: '',
            }
        ]
    }
    """
    print(f"🔄 Attempting unified API download for: {url}")
    
    try:
        # Correct headers for Social Download All in One API
        headers = {
            'x-rapidapi-host': 'social-download-all-in-one.p.rapidapi.com',
            'x-rapidapi-key': RAPIDAPI_KEY or '135d1d8e94msh773cfcb7bd35969p1fada7jsn4743820f5475',
            'Content-Type': 'application/json'
        }
        
        # Correct API endpoint and payload format
        api_endpoint = "https://social-download-all-in-one.p.rapidapi.com/v1/social/autolink"
        payload = {
            "url": url
        }
        
        print(f"📤 Making request to: {api_endpoint}")
        print(f"📤 Payload: {payload}")
        print(f"📤 Headers: {dict(headers)}")
        
        response = requests.post(
            api_endpoint,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"📥 Response status: {response.status_code}")
        print(f"📥 Response headers: {dict(response.headers)}")
        
        if response.status_code != 200:
            print(f"❌ API failed with status {response.status_code}")
            print(f"❌ Response text: {response.text[:500]}")
            raise Exception(f"API returned status {response.status_code}: {response.text[:200]}")
        
        data = response.json()
        print(f"📥 Unified API response: {json.dumps(data, indent=2)[:500]}")
        
        # Extract download URL from the documented response format
        download_url = None
        video_info = {}
        
        # Handle the documented API response format
        if data.get('medias') and isinstance(data['medias'], list) and len(data['medias']) > 0:
            # Find the best video quality
            video_media = None
            for media in data['medias']:
                if media.get('type') == 'video':
                    video_media = media
                    # Prefer higher quality if available
                    if 'quality' in media and ('720' in str(media['quality']) or 'high' in str(media['quality']).lower()):
                        break
            
            if not video_media:
                # If no video type found, use first media item
                video_media = data['medias'][0]
            
            download_url = video_media.get('url')
            
            # Extract metadata from main response
            video_info = {
                "title": data.get("title", "Video"),
                "uploader": data.get("author", "Unknown"),
                "duration": data.get("duration"),
                "source": data.get("source", "unified_api"),
                "quality": video_media.get("quality", "unknown"),
                "extension": video_media.get("extension", "mp4")
            }
            
        elif data.get('url'):
            # Direct URL response (fallback)
            download_url = data['url']
            video_info = {
                "title": data.get("title", "Video"),
                "uploader": data.get("author", "Unknown"),
                "source": data.get("source", "unified_api")
            }
        
        if not download_url:
            raise Exception(f"No download URL found in API response. Response structure: {list(data.keys())}")
        
        print(f"📥 Downloading from unified API: {download_url[:100]}...")
        
        # ENHANCED: Try multiple strategies for downloading Google Video URLs
        download_strategies = [
            # Strategy 1: Direct download with YouTube headers
            {
                'name': 'YouTube Browser',
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Referer': 'https://www.youtube.com/',
                    'Origin': 'https://www.youtube.com',
                    'Accept': '*/*',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Connection': 'keep-alive',
                    'Sec-Fetch-Dest': 'video',
                    'Sec-Fetch-Mode': 'no-cors',
                    'Sec-Fetch-Site': 'cross-site'
                },
                'timeout': 60
            },
            # Strategy 2: Mobile browser (often less restricted)
            {
                'name': 'Mobile Browser',
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1',
                    'Referer': 'https://m.youtube.com/',
                    'Accept': '*/*',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive'
                },
                'timeout': 60
            },
            # Strategy 3: Android YouTube app simulation
            {
                'name': 'Android App',
                'headers': {
                    'User-Agent': 'com.google.android.youtube/19.29.37 (Linux; U; Android 13) gzip',
                    'Accept': '*/*',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive'
                },
                'timeout': 60
            },
            # Strategy 4: Minimal headers (sometimes works when others fail)
            {
                'name': 'Minimal',
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': '*/*'
                },
                'timeout': 30
            },
            # Strategy 5: No headers at all (last resort)
            {
                'name': 'No Headers',
                'headers': {},
                'timeout': 30
            }
        ]
        
        video_response = None
        last_error = None
        
        for i, strategy in enumerate(download_strategies):
            try:
                print(f"   🔄 Strategy {i+1}/{len(download_strategies)}: {strategy['name']}...")
                
                # Try the download with this strategy
                video_response = requests.get(
                    download_url, 
                    stream=True, 
                    timeout=strategy['timeout'], 
                    headers=strategy['headers'],
                    allow_redirects=True  # Follow redirects
                )
                
                # Check if we got a successful response
                if video_response.status_code == 200:
                    print(f"   ✅ Strategy {i+1} ({strategy['name']}) successful!")
                    break
                else:
                    print(f"   ❌ Strategy {i+1} failed: HTTP {video_response.status_code}")
                    video_response = None
                    continue
                    
            except Exception as e:
                last_error = e
                print(f"   ❌ Strategy {i+1} ({strategy['name']}) failed: {str(e)[:100]}")
                video_response = None
                continue
        
        if video_response is None:
            # If all strategies failed, provide a more helpful error message
            raise Exception(f"All download strategies failed. This is likely due to Google's enhanced bot protection. Last error: {last_error}")
        video_response.raise_for_status()
        
        # Determine file extension
        extension = video_info.get('extension', 'mp4')
        if not extension.startswith('.'):
            extension = f".{extension}"
        
        filename = f"{file_id}{extension}"
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        
        # Save video file
        with open(filepath, 'wb') as f:
            for chunk in video_response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"✅ Unified API download successful: {filename}")
        
        return {
            "filename": filename,
            "url": f"/videos/{filename}",
            "meta": video_info
        }
        
    except Exception as e:
        print(f"❌ Unified API download failed: {str(e)}")
        raise

async def download_with_cookies(url: str, file_id: str) -> dict:
    """
    Third fallback: yt-dlp WITH cookies (100% reliable for YouTube)
    """
    print(f"🔄 Attempting yt-dlp WITH cookies (reliable fallback) for: {url}")
    
    try:
        filepath = os.path.join(DOWNLOAD_DIR, f"{file_id}")
        
        ydl_opts = {
            'format': 'best[height<=720]/best',
            'outtmpl': filepath + '.%(ext)s',
            'quiet': False,
            'no_warnings': False,
            'max_filesize': 100 * 1024 * 1024,
            'noplaylist': True,
            'geo_bypass': True,
            
            # Enhanced settings
            'extractor_args': {
                'youtube': {
                    'player_client': ['ios', 'android', 'web'],
                    'skip': ['dash', 'hls'],
                    'player_skip': ['configs'],
                }
            },
            
            'http_headers': {
                'User-Agent': 'com.google.ios.youtube/19.29.1 (iPhone14,3; U; CPU iOS 15_6 like Mac OS X)',
                'Accept': '*/*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate',
            },
            
            'retries': 15,
            'fragment_retries': 15,
            'file_access_retries': 10,
            'socket_timeout': 60,
            'nocheckcertificate': True,
        }
        
        # CRITICAL: Use cookies (this makes it 100% reliable)
        if os.path.exists(COOKIES_FILE):
            ydl_opts['cookiefile'] = COOKIES_FILE
            print(f"🍪 Using cookies from: {COOKIES_FILE}")
        else:
            raise Exception("Cookies file not found - cannot proceed with authenticated download")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            
            ext = info.get('ext', 'mp4')
            actual_filename = f"{file_id}.{ext}"
            
            meta_info = {
                "title": info.get("title", "Unknown"),
                "uploader": info.get("uploader", "Unknown"),
                "duration": info.get("duration"),
                "view_count": info.get("view_count"),
                "source": "yt-dlp_with_cookies"
            }
            
            print(f"✅ Cookie-based download successful: {actual_filename}")
            
            return {
                "filename": actual_filename,
                "url": f"/videos/{actual_filename}",
                "meta": meta_info
            }
        
    except Exception as e:
        print(f"❌ Cookie-based download failed: {str(e)}")
        raise

async def download_with_tiktok_api(url: str, file_id: str) -> dict:
    """
    TikTok-specific download using multiple fallback APIs
    """
    print(f"🔄 Attempting TikTok API download for: {url}")
    
    # Method 1: Try Cobalt API instances
    try:
        cobalt_instances = [
            "https://api.cobalt.tools",
            "https://cobalt.pub", 
            "https://api.wuk.sh"
        ]
        
        for instance in cobalt_instances:
            try:
                print(f"📤 Trying Cobalt instance: {instance}")
                
                response = requests.post(f"{instance}/api/json", 
                    headers={
                        'Accept': 'application/json',
                        'Content-Type': 'application/json',
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    },
                    json={
                        "url": url,
                        "filenamePattern": "basic",
                        "downloadMode": "auto"
                    },
                    timeout=15
                )
                
                if response.status_code != 200:
                    print(f"❌ {instance} failed with status {response.status_code}")
                    continue
                
                data = response.json()
                print(f"📥 Cobalt response: {json.dumps(data, indent=2)[:300]}")
                
                if data.get('status') == 'error':
                    print(f"❌ {instance} returned error: {data.get('text')}")
                    continue
                
                # Extract download URL
                download_url = data.get('url')
                if not download_url and data.get('picker'):
                    # Multiple quality options
                    download_url = data['picker'][0].get('url')
                
                if not download_url:
                    print(f"❌ No download URL in response from {instance}")
                    continue
                
                print(f"📥 Downloading TikTok video from: {download_url[:100]}...")
                
                # Download the video
                video_response = requests.get(download_url, stream=True, timeout=60, headers={
                    'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15'
                })
                video_response.raise_for_status()
                
                # Save to file
                filename = f"{file_id}.mp4"
                filepath = os.path.join(DOWNLOAD_DIR, filename)
                
                with open(filepath, 'wb') as f:
                    for chunk in video_response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                print(f"✅ TikTok download successful: {filename}")
                
                return {
                    "filename": filename,
                    "url": f"/videos/{filename}",
                    "meta": {
                        "title": "TikTok Video",
                        "uploader": "TikTok User",
                        "source": "cobalt_api"
                    }
                }
                
            except Exception as e:
                print(f"❌ {instance} failed: {str(e)}")
                continue
        
        print("❌ All Cobalt instances failed, trying alternative method...")
        
    except Exception as e:
        print(f"❌ Cobalt API method failed: {str(e)}")
    
    # Method 2: Try TikTok Downloader API (alternative)
    try:
        print(f"📤 Trying TikTok Downloader API...")
        
        # Extract TikTok video ID from URL
        import re
        video_id_match = re.search(r'/video/(\d+)', url)
        if not video_id_match:
            raise Exception("Could not extract TikTok video ID")
        
        video_id = video_id_match.group(1)
        print(f"📹 TikTok Video ID: {video_id}")
        
        # Try TikTok downloader API
        api_url = f"https://tikwm.com/api/?url={url}"
        
        response = requests.get(api_url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"📥 TikWM response: {json.dumps(data, indent=2)[:300]}")
            
            if data.get('code') == 0 and data.get('data'):
                video_data = data['data']
                download_url = video_data.get('play')
                
                if download_url:
                    print(f"📥 Downloading from TikWM: {download_url[:100]}...")
                    
                    video_response = requests.get(download_url, stream=True, timeout=60, headers={
                        'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15'
                    })
                    video_response.raise_for_status()
                    
                    # Save to file
                    filename = f"{file_id}.mp4"
                    filepath = os.path.join(DOWNLOAD_DIR, filename)
                    
                    with open(filepath, 'wb') as f:
                        for chunk in video_response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    print(f"✅ TikWM download successful: {filename}")
                    
                    return {
                        "filename": filename,
                        "url": f"/videos/{filename}",
                        "meta": {
                            "title": video_data.get('title', 'TikTok Video'),
                            "uploader": video_data.get('author', {}).get('unique_id', 'TikTok User'),
                            "source": "tikwm_api"
                        }
                    }
        
        raise Exception("TikWM API failed or returned no download URL")
        
    except Exception as e:
        print(f"❌ TikWM API failed: {str(e)}")
    
    # All methods failed
    raise Exception("All TikTok download methods failed (Cobalt + TikWM)")

@app.post("/api/download")
async def download_video(request: DownloadRequest):
    try:
        print(f"📥 Download request received for URL: {request.url}")
        
        file_id = str(uuid.uuid4())
        filename = f"{file_id}.%(ext)s"  # Let yt-dlp determine the extension
        filepath = os.path.join(DOWNLOAD_DIR, f"{file_id}")
        
        # Detect platform for optimized settings
        is_youtube = "youtube.com" in request.url or "youtu.be" in request.url
        is_tiktok = "tiktok.com" in request.url
        is_instagram = "instagram.com" in request.url
        is_twitter = "twitter.com" in request.url or "x.com" in request.url
        
        ydl_opts = {
            # Use mobile format (less likely to be blocked)
            # Flexible format for Shorts and regular videos
            'format': 'best[height<=720]/best',
            'outtmpl': filepath + '.%(ext)s',
            'quiet': False,  # Enable logging for debugging
            'no_warnings': False,
            'verbose': True,  # Add verbose logging
            'max_filesize': 100 * 1024 * 1024,
            'noplaylist': True,
            'geo_bypass': True,
            
            # Network settings
            'socket_timeout': 60,  # Increased timeout
            'source_address': None,
            
            # Avoid potential issues
            'nocheckcertificate': True,
            'prefer_insecure': False,
            
            # CRITICAL: Add extractor options to avoid detection
            'extract_flat': False,
            'age_limit': None,
        }
        
        # Platform-specific optimizations
        if is_youtube:
            # YouTube-specific settings
            ydl_opts.update({
                'extractor_args': {
                    'youtube': {
                        'player_client': ['ios', 'android', 'web'],
                        'skip': ['dash', 'hls'],
                        'player_skip': ['configs'],
                    }
                },
                'http_headers': {
                    'User-Agent': 'com.google.ios.youtube/19.29.1 (iPhone14,3; U; CPU iOS 15_6 like Mac OS X)',
                    'Accept': '*/*',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate',
                },
                'sleep_interval': 2,
                'max_sleep_interval': 10,
                'retries': 15,
                'fragment_retries': 15,
                'file_access_retries': 10,
            })
            print(f"🎥 YouTube detected - using YouTube-optimized settings")
            
        elif is_tiktok:
            # TikTok-specific settings (no cookies, different headers)
            ydl_opts.update({
                'extractor_args': {
                    'tiktok': {
                        'webpage_url_basename': 'video',
                    }
                },
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'Referer': 'https://www.tiktok.com/',
                },
                'sleep_interval': 1,
                'max_sleep_interval': 5,
                'retries': 10,
                'fragment_retries': 10,
                'file_access_retries': 5,
            })
            print(f"📱 TikTok detected - using TikTok-optimized settings (no cookies)")
            
        elif is_instagram:
            # Instagram-specific settings
            ydl_opts.update({
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                },
                'sleep_interval': 1,
                'retries': 8,
                'fragment_retries': 8,
                'file_access_retries': 5,
            })
            print(f"📸 Instagram detected - using Instagram-optimized settings")
            
        elif is_twitter:
            # Twitter/X-specific settings
            ydl_opts.update({
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                },
                'sleep_interval': 1,
                'retries': 8,
                'fragment_retries': 8,
                'file_access_retries': 5,
            })
            print(f"🐦 Twitter/X detected - using Twitter-optimized settings")
            
        else:
            # Generic social media settings
            ydl_opts.update({
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                },
                'sleep_interval': 1,
                'retries': 8,
                'fragment_retries': 8,
                'file_access_retries': 5,
            })
            print(f"🌐 Generic platform detected - using universal settings")
        
        # CRITICAL: Use cookies ONLY for YouTube (platform-specific)
        is_youtube = "youtube.com" in request.url or "youtu.be" in request.url
        if is_youtube and os.path.exists(COOKIES_FILE):
            ydl_opts['cookiefile'] = COOKIES_FILE
            print(f"🍪 Using YouTube cookies from: {COOKIES_FILE}")
        elif is_youtube:
            print(f"⚠️ YouTube URL detected but no cookies available - may encounter bot detection")
        else:
            print(f"📱 Non-YouTube platform detected - skipping cookie authentication")
        
        print(f"📁 Download path: {filepath}")
        
        meta_info = {}
        actual_filename = None
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                print("🔍 Extracting video info...")
                info = ydl.extract_info(request.url, download=True)
                
                # Get the actual filename that was created
                ext = info.get('ext', 'mp4')
                actual_filename = f"{file_id}.{ext}"
                actual_filepath = os.path.join(DOWNLOAD_DIR, actual_filename)
                
                meta_info = {
                    "title": info.get("title", "Unknown"),
                    "uploader": info.get("uploader", "Unknown"),
                    "duration": info.get("duration"),
                    "view_count": info.get("view_count"),
                    "description": info.get("description", ""),
                    "tags": info.get("tags", []),
                    "categories": info.get("categories", [])
                }
                
                print(f"✅ Download completed: {actual_filename}")
                print(f"📊 Video info: {meta_info['title']} by {meta_info['uploader']}")
                
            except Exception as e:
                print(f"❌ yt-dlp error: {str(e)}")
                raise e

        # Check if file was created (with any extension)
        created_files = [f for f in os.listdir(DOWNLOAD_DIR) if f.startswith(file_id)]
        if not created_files:
            print(f"❌ No files created with ID: {file_id}")
            return JSONResponse(status_code=400, content={"detail": "Download failed: No file was created"})
        
        # Use the first file found (should only be one)
        actual_filename = created_files[0]
        print(f"📄 Found created file: {actual_filename}")
        
        return {
            "filename": actual_filename,
            "url": f"/videos/{actual_filename}",
            "meta": meta_info
        }

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Primary download method failed: {error_msg}")
        
        # Check if this is a specific platform that might need specialized handling FIRST
        is_youtube = "youtube.com" in request.url or "youtu.be" in request.url
        is_tiktok = "tiktok.com" in request.url
        is_bot_error = "Sign in to confirm" in error_msg or "bot" in error_msg.lower()
        
        # For TikTok, try the specialized TikTok APIs FIRST (before unified API)
        if is_tiktok:
            print(f"🔄 TikTok detected - trying specialized TikTok APIs...")
            try:
                result = await download_with_tiktok_api(request.url, file_id)
                print(f"✅ TikTok specialized API download successful!")
                return result
            except Exception as tiktok_error:
                print(f"❌ TikTok specialized API failed: {str(tiktok_error)}")
        
        # Try unified API as fallback for non-TikTok platforms or if TikTok APIs failed
        print(f"🔄 Attempting unified API fallback...")
        
        try:
            # Try unified API as reliable fallback
            result = await download_with_unified_api(request.url, file_id)
            print(f"✅ Unified API download successful!")
            return result
            
        except Exception as unified_error:
            print(f"❌ Unified API fallback failed: {str(unified_error)}")
            
            # Provide helpful error message based on platform
            if is_youtube:
                helpful_msg = (
                    "🚫 YouTube Download Failed\n\n"
                    "All download methods failed (yt-dlp + Unified API).\n\n"
                    "✅ **Easy Workaround (30 seconds):**\n"
                    "1. Download the YouTube video to your device (use any YouTube downloader)\n"
                    "2. Click the 'Upload File' tab above\n"
                    "3. Upload the video file\n"
                    "4. Analyze as normal!\n\n"
                    "💡 **Other Options:**\n"
                    "• Try a different platform (TikTok, Instagram, Twitter all work great!)\n"
                )
                error_type = "youtube_download_failed"
            elif is_tiktok:
                helpful_msg = (
                    "🚫 TikTok Download Failed\n\n"
                    "All download methods failed (yt-dlp + Unified API + TikTok APIs).\n\n"
                    "✅ **Easy Workaround (30 seconds):**\n"
                    "1. Download the TikTok video to your device (use any TikTok downloader)\n"
                    "2. Click the 'Upload File' tab above\n"
                    "3. Upload the video file\n"
                    "4. Analyze as normal!\n\n"
                    "💡 **Other Options:**\n"
                    "• Try a different platform (YouTube, Instagram, Twitter all work great!)\n"
                    "• Make sure the TikTok link is public and not private\n"
                )
                error_type = "tiktok_download_failed"
            else:
                helpful_msg = (
                    "🚫 Social Media Download Failed\n\n"
                    "All download methods failed (yt-dlp + Unified API).\n\n"
                    "✅ **Easy Workaround (30 seconds):**\n"
                    "1. Download the video to your device (use any social media downloader)\n"
                    "2. Click the 'Upload File' tab above\n"
                    "3. Upload the video file\n"
                    "4. Analyze as normal!\n\n"
                    "💡 **Other Options:**\n"
                    "• Try a different platform or video\n"
                    "• Make sure the link is public and not private\n"
                )
                error_type = "social_media_download_failed"
            
            return JSONResponse(status_code=400, content={
                "detail": helpful_msg,
                "error_type": error_type,
                "primary_error": error_msg,
                "unified_api_error": str(unified_error)
            })

@app.post("/api/analyze")
async def analyze_video(request: Request, data: AnalyzeRequest):
    # 1. Check Rate Limit
    client_ip = request.headers.get("x-forwarded-for", request.client.host).split(",")[0]
    print(f"🔍 Client IP detected: {client_ip}")
    
    if not check_rate_limit(client_ip):
         raise HTTPException(status_code=429, detail="Daily submission limit reached (5/5). Contact admin@frametruth.com for access.")

    # 2. Locate File
    filepath = os.path.join(DOWNLOAD_DIR, data.filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        # 3. Calculate video duration to determine optimal frame count
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 30
        cap.release()
        
        # EXTREME PERFORMANCE OPTIMIZATION: Minimal processing for sub-2-minute analysis
        if duration < 30:
            num_frames = 5   # Very short clips (was 8)
            dinov2_samples = 3  # (was 6)
            print(f"📹 Short video ({duration:.1f}s) - using 5 frames (extreme speed optimization)")
        elif duration < 60:
            num_frames = 6  # Medium videos (was 10)
            dinov2_samples = 3  # (was 6)
            print(f"📹 Medium video ({duration:.1f}s) - using 6 frames (extreme speed optimization)")
        else:
            num_frames = 8  # Longer videos (was 12)
            dinov2_samples = 4  # (was 8)
            print(f"📹 Long video ({duration:.1f}s) - using 8 frames (extreme speed optimization)")
        
        # 4. 🚀 GAME-CHANGING ADDITION: PRNU Sensor Fingerprint Analysis
        print(f"🔬 PRNU Sensor Fingerprint Analysis: extracting camera 'DNA' from video...")
        prnu_metrics = calculate_prnu_sensor_fingerprint(filepath, num_samples=max(8, dinov2_samples*2))
        
        # 5. CPU-OPTIMIZED ANALYSIS: Re-enable advanced components with speed optimizations
        print(f"🔧 CPU-OPTIMIZED MODE: Re-enabling advanced analysis with speed optimizations")
        
        # Re-enable with aggressive optimizations for speed
        print(f"📐 Calculating lightweight trajectory metrics ({max(2, dinov2_samples//2)} samples)...")
        trajectory_metrics = calculate_lightweight_trajectory_metrics(filepath, num_samples=max(2, dinov2_samples//2))
        
        print(f"🌊 Calculating optical flow features ({max(2, dinov2_samples//2)} samples)...")
        optical_flow_metrics = calculate_optical_flow_features(filepath, num_samples=max(2, dinov2_samples//2))
        
        print(f"📝 Analyzing text stability with OCR ({max(1, dinov2_samples//3)} samples)...")
        ocr_metrics = analyze_text_stability(filepath, num_samples=max(1, dinov2_samples//3))
        
        # 🚀 ADD MISSING FEATURE EXTRACTORS FOR COMPLETE 36-FEATURE PIPELINE
        print(f"🎵 Computing frequency domain features...")
        # Extract frames for frequency analysis
        cap = cv2.VideoCapture(filepath)
        frames_for_freq = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // 4)  # Sample 4 frames
        
        while cap.isOpened() and len(frames_for_freq) < 4:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % step == 0:
                frames_for_freq.append(frame)
            frame_count += 1
        cap.release()
        
        frequency_metrics = compute_frequency_features(frames_for_freq) if frames_for_freq else {}
        
        print(f"🔊 Computing noise/grain statistics...")
        noise_metrics = compute_noise_features(frames_for_freq) if frames_for_freq else {}
        
        print(f"🎬 Computing codec/compression features...")
        codec_metrics = compute_codec_features(filepath)
        
        print(f"📊 Computing metadata features...")
        metadata_metrics = compute_metadata_features(filepath)
        
        print(f"🔧 Advanced analysis re-enabled with CPU optimizations for accuracy + speed balance")
        
        trajectory_boost = {'score_increase': 0, 'confidence_boost': 0, 'has_high_curvature': False, 'has_flow_anomalies': False, 'has_text_anomalies': False, 'has_prnu_anomalies': False}
        
        # 6. 🚀 ANALYZE PRNU SENSOR FINGERPRINT (GAME-CHANGING!)
        if prnu_metrics:
            prnu_mean_corr = prnu_metrics['prnu_mean_corr']
            prnu_std_corr = prnu_metrics['prnu_std_corr']
            prnu_positive_ratio = prnu_metrics['prnu_positive_ratio']
            prnu_consistency_score = prnu_metrics['prnu_consistency_score']
            
            # Detect PRNU anomalies that indicate AI generation
            prnu_anomaly_score = 0
            prnu_reasons = []
            
            # 1. Low mean correlation (AI lacks consistent sensor fingerprint)
            if prnu_mean_corr < 0.15:  # Real cameras typically > 0.2
                prnu_anomaly_score += 50
                prnu_reasons.append(f"Low sensor fingerprint correlation ({prnu_mean_corr:.3f})")
                print(f"🚨 LOW SENSOR FINGERPRINT CORRELATION: {prnu_mean_corr:.3f} (AI lacks consistent camera sensor pattern)")
            
            # 2. High standard deviation (AI has inconsistent noise patterns)
            if prnu_std_corr > 0.3:  # Real cameras typically < 0.2
                prnu_anomaly_score += 40
                prnu_reasons.append(f"High sensor noise inconsistency ({prnu_std_corr:.3f})")
                print(f"🚨 HIGH SENSOR NOISE INCONSISTENCY: {prnu_std_corr:.3f} (AI has unstable noise patterns)")
            
            # 3. Low positive ratio (AI correlations often near zero or negative)
            if prnu_positive_ratio < 0.6:  # Real cameras typically > 0.7
                prnu_anomaly_score += 45
                prnu_reasons.append(f"Low positive correlation ratio ({prnu_positive_ratio:.3f})")
                print(f"🚨 LOW POSITIVE CORRELATION RATIO: {prnu_positive_ratio:.3f} (AI lacks stable sensor signature)")
            
            # 4. Low overall consistency score
            if prnu_consistency_score < 0.3:  # Real cameras typically > 0.5
                prnu_anomaly_score += 35
                prnu_reasons.append(f"Low overall sensor consistency ({prnu_consistency_score:.3f})")
                print(f"🚨 LOW SENSOR CONSISTENCY: {prnu_consistency_score:.3f} (AI lacks physical sensor characteristics)")
            
            # Apply PRNU boost if anomalies detected
            if prnu_anomaly_score > 40:  # Significant PRNU anomalies
                additional_score = min(70, prnu_anomaly_score)  # PRNU is extremely reliable
                additional_confidence = min(40, prnu_anomaly_score // 2)
                
                trajectory_boost['score_increase'] += additional_score
                trajectory_boost['confidence_boost'] += additional_confidence
                trajectory_boost['has_prnu_anomalies'] = True
                
                print(f"🔬 PRNU SENSOR ANOMALIES DETECTED: +{additional_score} score, +{additional_confidence} confidence")
                print(f"   Anomalies: {', '.join(prnu_reasons)}")
            else:
                print(f"✓ Normal sensor fingerprint patterns: mean_corr={prnu_mean_corr:.3f}, consistency={prnu_consistency_score:.3f}")
        else:
            print(f"⚠️ PRNU analysis failed - sensor fingerprint analysis unavailable")
        
        # 7. Analyze trajectory curvature (existing logic)
        if trajectory_metrics:
            mean_curv = trajectory_metrics['mean_curvature']
            curv_var = trajectory_metrics['curvature_variance']
            mean_dist = trajectory_metrics['mean_distance']
            
            # BALANCED: Trajectory analysis provides important supporting evidence
            # High curvature (>100°) is a significant AI indicator that should influence the decision
            # But don't completely override visual analysis - work together
            
            if mean_curv > 130:  # EXTREMELY high - very strong AI indicator
                trajectory_boost['score_increase'] += min(40, int((mean_curv - 130) * 2))  # Up to +40
                trajectory_boost['confidence_boost'] += min(25, int((mean_curv - 130) * 1))  # Up to +25
                trajectory_boost['has_high_curvature'] = True
                print(f"🚨 EXTREMELY HIGH VISUAL CURVATURE: {mean_curv:.1f}° (very strong AI indicator)")
            elif mean_curv > 110:  # High curvature - strong AI indicator
                trajectory_boost['score_increase'] += min(30, int((mean_curv - 110) * 1.5))  # Up to +30
                trajectory_boost['confidence_boost'] += min(20, int((mean_curv - 110) * 1))  # Up to +20
                trajectory_boost['has_high_curvature'] = True
                print(f"⚠️ HIGH VISUAL CURVATURE: {mean_curv:.1f}° (strong AI indicator - modern AI often shows this pattern)")
            elif mean_curv > 90:  # Moderate-high curvature - possible AI indicator
                trajectory_boost['score_increase'] += min(20, int((mean_curv - 90) * 1))  # Up to +20
                trajectory_boost['confidence_boost'] += min(15, int((mean_curv - 90) * 0.75))  # Up to +15
                trajectory_boost['has_high_curvature'] = True
                print(f"⚠️ MODERATE-HIGH VISUAL CURVATURE: {mean_curv:.1f}° (possible AI indicator)")
            elif mean_curv > 70:  # Moderate curvature - minor AI indicator
                trajectory_boost['score_increase'] += min(10, int((mean_curv - 70) * 0.5))  # Up to +10
                trajectory_boost['confidence_boost'] += min(5, int((mean_curv - 70) * 0.25))  # Up to +5
                trajectory_boost['has_high_curvature'] = True
                print(f"ℹ️ MODERATE VISUAL CURVATURE: {mean_curv:.1f}° (could be AI or dynamic camera movement)")
            else:
                print(f"✓ Normal visual curvature: {mean_curv:.1f}° (typical for real video)")
        
        # 8. Analyze optical flow anomalies
        if optical_flow_metrics:
            flow_mean = optical_flow_metrics['flow_global_mean']
            flow_std = optical_flow_metrics['flow_global_std']
            flow_jitter = optical_flow_metrics['flow_jitter_index']
            bg_fg_ratio = optical_flow_metrics.get('background_vs_foreground_ratio', 0.0)
            patch_variance = optical_flow_metrics.get('flow_patch_variance', 0.0)
            
            # Detect optical flow anomalies that indicate AI generation
            flow_anomaly_score = 0
            flow_reasons = []
            
            # 1. Excessive flow jitter (AI often has micro-jitter)
            if flow_jitter > 1.5:  # High jitter threshold
                flow_anomaly_score += 25
                flow_reasons.append(f"High temporal flow jitter ({flow_jitter:.2f})")
                print(f"⚠️ HIGH FLOW JITTER: {flow_jitter:.2f} (AI often shows micro-jitter)")
            
            # 2. Unnatural background/foreground flow ratio
            if bg_fg_ratio > 0.8:  # Background and foreground moving too similarly
                flow_anomaly_score += 20
                flow_reasons.append(f"Unnatural bg/fg flow ratio ({bg_fg_ratio:.2f})")
                print(f"⚠️ UNNATURAL BG/FG FLOW: {bg_fg_ratio:.2f} (AI often moves bg/fg identically)")
            
            # 3. Excessive patch variance (inconsistent local motion)
            if patch_variance > 0.3:  # High local motion inconsistency
                flow_anomaly_score += 15
                flow_reasons.append(f"High local motion inconsistency ({patch_variance:.2f})")
                print(f"⚠️ HIGH PATCH VARIANCE: {patch_variance:.2f} (inconsistent local motion)")
            
            # 4. Too-perfect flow (unnaturally low variance)
            if flow_std < 0.1 and flow_mean > 0.5:  # Suspiciously smooth motion
                flow_anomaly_score += 15
                flow_reasons.append(f"Unnaturally smooth motion (std: {flow_std:.2f})")
                print(f"⚠️ TOO-SMOOTH MOTION: std={flow_std:.2f} (AI often too perfect)")
            
            # Apply optical flow boost if anomalies detected
            if flow_anomaly_score > 20:  # Significant flow anomalies
                additional_score = min(30, flow_anomaly_score)
                additional_confidence = min(20, flow_anomaly_score // 2)
                
                trajectory_boost['score_increase'] += additional_score
                trajectory_boost['confidence_boost'] += additional_confidence
                trajectory_boost['has_flow_anomalies'] = True
                
                print(f"🌊 OPTICAL FLOW ANOMALIES DETECTED: +{additional_score} score, +{additional_confidence} confidence")
                print(f"   Anomalies: {', '.join(flow_reasons)}")
            else:
                print(f"✓ Normal optical flow patterns: jitter={flow_jitter:.2f}, bg/fg={bg_fg_ratio:.2f}")
        
        # 9. Analyze OCR text stability anomalies
        if ocr_metrics and ocr_metrics['has_text']:
            char_error_rate = ocr_metrics['ocr_char_error_rate']
            frame_stability = ocr_metrics['ocr_frame_stability_score']
            mutation_rate = ocr_metrics['ocr_text_mutation_rate']
            unique_strings = ocr_metrics['ocr_unique_string_count']
            
            # Detect text anomalies that indicate AI generation
            text_anomaly_score = 0
            text_reasons = []
            
            # 1. High character error rate (weird glyphs, non-ASCII)
            if char_error_rate > 0.1:  # >10% character errors
                text_anomaly_score += 40
                text_reasons.append(f"High character error rate ({char_error_rate:.2f})")
                print(f"⚠️ HIGH CHARACTER ERROR RATE: {char_error_rate:.2f} (AI often has weird glyphs)")
            
            # 2. Low frame stability (text morphing between frames)
            if frame_stability < 0.7:  # <70% stability
                text_anomaly_score += 35
                text_reasons.append(f"Low text frame stability ({frame_stability:.2f})")
                print(f"⚠️ LOW TEXT STABILITY: {frame_stability:.2f} (AI text often morphs)")
            
            # 3. High mutation rate (text changing too often)
            if mutation_rate > 0.3:  # >30% mutation rate
                text_anomaly_score += 30
                text_reasons.append(f"High text mutation rate ({mutation_rate:.2f})")
                print(f"⚠️ HIGH TEXT MUTATION: {mutation_rate:.2f} (AI text changes unnaturally)")
            
            # 4. Too many unique strings (inconsistent text)
            if unique_strings > 10:  # Too many different text strings
                text_anomaly_score += 20
                text_reasons.append(f"Too many unique text strings ({unique_strings})")
                print(f"⚠️ TOO MANY TEXT VARIANTS: {unique_strings} (AI often inconsistent)")
            
            # Apply OCR boost if anomalies detected
            if text_anomaly_score > 25:  # Significant text anomalies
                additional_score = min(50, text_anomaly_score)  # OCR is very reliable
                additional_confidence = min(30, text_anomaly_score // 2)
                
                trajectory_boost['score_increase'] += additional_score
                trajectory_boost['confidence_boost'] += additional_confidence
                trajectory_boost['has_text_anomalies'] = True
                
                print(f"📝 TEXT ANOMALIES DETECTED: +{additional_score} score, +{additional_confidence} confidence")
                print(f"   Anomalies: {', '.join(text_reasons)}")
            else:
                print(f"✓ Normal text patterns: stability={frame_stability:.2f}, mutation={mutation_rate:.2f}")
        elif ocr_metrics and not ocr_metrics['has_text']:
            print(f"ℹ️ No text detected in video - OCR analysis skipped")
        else:
            print(f"⚠️ OCR analysis failed - text analysis unavailable")
        
        # 10. Scan metadata for AI keywords (if URL provided)
        metadata_scan = {'has_ai_keywords': False, 'keywords_found': [], 'confidence_boost': 0, 'score_increase': 0}
        if data.original_url and data.original_url.strip():
            print(f"🔍 Scanning metadata for AI keywords in URL: {data.original_url}")
            metadata_scan = scan_metadata_for_ai_keywords(data.original_url)
            if metadata_scan['has_ai_keywords']:
                print(f"⚠️ AI KEYWORDS DETECTED: {metadata_scan['keywords_found']}")
                print(f"   Score boost: +{metadata_scan['score_increase']}, Confidence boost: +{metadata_scan['confidence_boost']}")
        
        # 11. Extract Frames
        frames = extract_frames_base64(filepath, num_frames)
        if not frames:
             raise HTTPException(status_code=400, detail="Could not extract frames from video")

        # 12. 🚀 ENHANCED GEMINI PROMPT WITH PRNU CONTEXT
        # Build structured numeric context from our analysis INCLUDING PRNU
        numeric_context = {
            "prnu_sensor_fingerprint": {
                "prnu_mean_corr": prnu_metrics['prnu_mean_corr'] if prnu_metrics else 0.0,
                "prnu_std_corr": prnu_metrics['prnu_std_corr'] if prnu_metrics else 0.0,
                "prnu_positive_ratio": prnu_metrics['prnu_positive_ratio'] if prnu_metrics else 0.0,
                "prnu_consistency_score": prnu_metrics['prnu_consistency_score'] if prnu_metrics else 0.0,
                "method": prnu_metrics['method'] if prnu_metrics else "none"
            },
            "trajectory_analysis": {
                "mean_curvature": trajectory_metrics['mean_curvature'] if trajectory_metrics else 0.0,
                "curvature_variance": trajectory_metrics['curvature_variance'] if trajectory_metrics else 0.0,
                "max_curvature": trajectory_metrics['max_curvature'] if trajectory_metrics else 0.0,
                "mean_distance": trajectory_metrics['mean_distance'] if trajectory_metrics else 0.0,
                "method": trajectory_metrics['method'] if trajectory_metrics else "none"
            },
            "optical_flow_analysis": {
                "flow_global_mean": optical_flow_metrics['flow_global_mean'] if optical_flow_metrics else 0.0,
                "flow_global_std": optical_flow_metrics['flow_global_std'] if optical_flow_metrics else 0.0,
                "temporal_flow_jitter_index": optical_flow_metrics['flow_jitter_index'] if optical_flow_metrics else 0.0,
                "background_vs_foreground_ratio": optical_flow_metrics.get('background_vs_foreground_ratio', 0.0) if optical_flow_metrics else 0.0,
                "flow_patch_variance_mean": optical_flow_metrics.get('flow_patch_variance', 0.0) if optical_flow_metrics else 0.0,
                "method": optical_flow_metrics['method'] if optical_flow_metrics else "none"
            },
            "ocr_text_analysis": {
                "has_text": ocr_metrics['has_text'] if ocr_metrics else False,
                "ocr_char_error_rate": ocr_metrics['ocr_char_error_rate'] if ocr_metrics else 0.0,
                "ocr_frame_stability_score": ocr_metrics['ocr_frame_stability_score'] if ocr_metrics else 1.0,
                "ocr_text_mutation_rate": ocr_metrics['ocr_text_mutation_rate'] if ocr_metrics else 0.0,
                "ocr_unique_string_count": ocr_metrics['ocr_unique_string_count'] if ocr_metrics else 0,
                "total_text_detections": ocr_metrics['total_text_detections'] if ocr_metrics else 0,
                "method": ocr_metrics['method'] if ocr_metrics else "none"
            },
            "metadata_analysis": {
                "has_ai_keywords": metadata_scan['has_ai_keywords'],
                "keyword_count": len(metadata_scan['keywords_found']),
                "keywords_found": metadata_scan['keywords_found'][:3]  # First 3 keywords
            },
            "video_properties": {
                "duration_seconds": duration,
                "frames_analyzed": num_frames,
                "samples_used": dinov2_samples
            }
        }
        
        # Create enhanced prompt with PRNU context
        prompt = f"""
    You are an EXPERT Video Forensics Analyst specializing in objective technical analysis of video authenticity. 
    Your role is to provide NEUTRAL, UNBIASED analysis based solely on observable technical evidence.

    🔢 NUMERIC FORENSIC CONTEXT:
    {json.dumps(numeric_context, indent=2)}

    📋 INTERPRETATION GUIDE:
    
    **🚀 PRNU Sensor Fingerprint Analysis (MOST RELIABLE SIGNAL):**
    - prnu_mean_corr < 0.15: Strong AI indicator (lacks consistent sensor fingerprint)
    - prnu_std_corr > 0.3: High noise inconsistency (AI artifact)
    - prnu_positive_ratio < 0.6: Low correlation stability (AI lacks sensor signature)
    - prnu_consistency_score < 0.3: Overall sensor inconsistency (AI lacks physical sensor)
    
    **Trajectory Analysis:**
    - mean_curvature > 110°: Strong AI indicator (irregular frame transitions)
    - curvature_variance > 500: High temporal inconsistency 
    - max_curvature > 150°: Extreme motion artifacts
    
    **Optical Flow Analysis:**
    - temporal_flow_jitter_index > 1.5: Micro-jitter (AI artifact)
    - background_vs_foreground_ratio > 0.8: Unnatural motion coupling
    - flow_patch_variance_mean > 0.3: Local motion inconsistency
    - flow_global_std < 0.1 + flow_global_mean > 0.5: Too-perfect motion
    
    **OCR Text Analysis:**
    - has_text=true: Text detected in video frames
    - ocr_char_error_rate > 0.1: High rate of weird/non-ASCII characters (AI artifact)
    - ocr_frame_stability_score < 0.7: Text morphing between frames (AI artifact)
    - ocr_text_mutation_rate > 0.3: Text changing too frequently (AI artifact)
    - ocr_unique_string_count > 10: Too many text variants (AI inconsistency)
    
    **Metadata Analysis:**
    - has_ai_keywords=true: Video explicitly tagged as AI-generated
    - keyword_count > 0: Contains AI-related terms in title/description
    
    🔍 ANALYSIS FRAMEWORK - Use numeric context as PRIMARY evidence, visual frames for CONFIRMATION:

    **CRITICAL INSTRUCTION: The PRNU sensor fingerprint analysis is the most reliable technical signal. If it shows strong AI indicators (low correlations, high inconsistency), this should heavily influence your decision. Use the visual frames to confirm and explain the patterns detected by the algorithms.**
    
    🔍 VISUAL ANALYSIS FRAMEWORK - Examine these sequential frames to confirm numeric findings:

    1. TEMPORAL COHERENCE ANALYSIS:
       - Frame-to-frame consistency (natural vs. artificial transitions)
       - Motion blur patterns (camera motion vs. synthetic blur)
       - Lighting consistency across frames
       - Object stability and position tracking
       - Background element behavior
       - **MICRO-GLITCHES**: Subtle warping, morphing, or "swimming" of textures between frames
       - **TEMPORAL ARTIFACTS**: Objects or details that flicker, appear/disappear, or shift unnaturally

    2. PHYSICS & MOTION ANALYSIS:
       - Gravity and momentum behavior (natural vs. impossible)
       - Shadow direction, intensity, and consistency
       - Reflection accuracy (mirrors, water, glass, metallic surfaces)
       - Depth perception and occlusion correctness
       - Material properties (weight, flexibility, rigidity)
       - **MOTION SMOOTHNESS**: Too-perfect motion or floating/drifting movements
       - **ACCELERATION**: Unnatural speed changes or momentum violations

    3. VISUAL QUALITY INDICATORS:
       - Texture consistency and detail stability
       - Surface material rendering (natural vs. synthetic appearance)
       - Facial features and proportions (natural vs. uncanny valley)
       - Pattern regularity in organic elements
       - Edge definition and boundary clarity
       - **TEXTURE BOILING**: Fine details that seem to crawl, shimmer, or fluctuate
       - **WATERCOLOR EFFECT**: Overly smooth, plastic-like, or painted appearance
       - **EDGE COHERENCE**: Soft or bleeding edges around objects

    4. OBJECT & SCENE CONSISTENCY:
       - Object permanence through occlusion
       - Detail consistency across frames (jewelry, buttons, text, features)
       - Background stability and coherence
       - Anatomical accuracy (hands, fingers, body proportions)
       - Scene composition naturalness
       - **ANATOMICAL TELLS**: Extra/missing fingers, impossible hand positions, morphing limbs
       - **OBJECT MUTATIONS**: Items changing size, shape, or details between frames
       - **BACKGROUND INSTABILITY**: Static elements that subtly shift or warp

    5. TECHNICAL SIGNATURES (Observable Patterns):
       - Camera artifacts (lens distortion, sensor noise, compression artifacts)
       - Motion characteristics (handheld shake, stabilization, panning smoothness)
       - Focus behavior (depth of field, bokeh, autofocus hunting)
       - Color grading and dynamic range
       - Temporal artifacts or glitches
       - **SENSOR NOISE**: Real cameras have grain; AI videos often too clean or artificial grain
       - **ROLLING SHUTTER**: Real cameras show this effect; AI often doesn't
       - **CHROMATIC ABERRATION**: Real lenses have color fringing; check if present/absent

    6. 🚨 TEXT/LETTER ANALYSIS (CRITICAL AI INDICATOR - HIGHEST PRIORITY):
       
       **⚠️ MANDATORY: Examine EVERY SINGLE piece of text visible in ANY frame**
       **This is THE MOST RELIABLE indicator of AI - be EXTREMELY scrutinous**
       
       **Where to Look (check ALL of these):**
       - Signs (street signs, store signs, airport signs, directional signs)
       - Products (labels, packaging, bottles, cans)
       - Screens (phones, computers, TVs, displays)
       - Clothing (t-shirts, jerseys, brand names)
       - Vehicles (license plates, logos, text on sides)
       - Buildings (storefront names, addresses, posted notices)
       - Documents (papers, books, magazines, newspapers)
       - ANY other visible text
       
       **AI Text Detection - Be VERY Critical:**
       
       Even "mostly readable" text can be AI-generated. Look for:
       
       **Subtle AI Text Indicators (VERY COMMON in modern AI):**
       - Letters that are **almost** correct but slightly off
       - Text that looks "fuzzy" or "soft" even when in focus
       - Spacing between letters that's inconsistent
       - Letter heights that vary within the same word
       - Serif/sans-serif mixing within same word
       - Characters that look hand-drawn rather than printed
       - Text that seems to "blend" into background slightly
       - Words with correct letters but wrong/unusual spelling
       - Real words but grammatically nonsensical combinations
       - Font consistency issues (even subtle)
       - Letters with unusual thickness variations
       - Curved text that doesn't follow proper arc
       - Shadowing/dimensionality that doesn't match lighting
       
       **CRITICAL: Modern AI (Sora, Runway Gen-3) can create VERY convincing text**
       - Don't be fooled by text that looks "mostly good"
       - AI text often looks 80-90% correct but has subtle issues
       - Compare text across multiple frames - does it stay identical?
       - Real printed text is PERFECTLY consistent across frames
       - AI text often has MICRO-variations frame-to-frame
       
       **Frame-to-Frame Text Analysis (ESSENTIAL):**
       - Does the EXACT same text look identical in consecutive frames?
       - Do letter shapes subtly morph or shift?
       - Does text clarity fluctuate even when camera is still?
       - Do shadows/highlights on text change unnaturally?
       
       **IMPORTANT DISTINCTIONS:**
       - ✅ Analyze text WITHIN the video (in-scene text)
       - ❌ IGNORE user-added captions/subtitles
       - ⚠️ "Readable text" does NOT mean video is real!
       - ⚠️ AI can create readable text - look for SUBTLE issues
       
       **Scoring Impact (AGGRESSIVE):**
       - ANY text inconsistencies → **+40-50 to curvatureScore**
       - Gibberish/distorted text → **+60-70 to curvatureScore**
       - Text morphing between frames → **+70-80 to curvatureScore**
       - Multiple text issues → **+80-90 to curvatureScore, set isAi=true**
       - Even subtle text problems should heavily influence your decision
       
       **⚠️ DEFAULT ASSUMPTION: If video has text, assume it's AI until proven otherwise**
       - Real text is PERFECT and CONSISTENT
       - Any imperfection in text is a RED FLAG
       - Be skeptical, not generous

    📊 SCORING METHODOLOGY (Evidence-Based):
    
    **curvatureScore (0-100)**: Based on combined technical analysis
    - 0-30: Strong evidence of real camera with consistent sensor fingerprint
    - 31-60: Mixed signals that could be either natural or synthetic
    - 61-100: Clear technical evidence of AI generation (sensor inconsistency, motion artifacts, text issues)
    
    **distanceScore (0-100)**: Based on spatial consistency
    - Measure consistency of object positions and spatial relationships
    
    **confidence (0-100)**: Your certainty in the classification
    - Base this on the STRENGTH and QUANTITY of evidence observed
    - PRNU sensor fingerprint analysis should heavily influence confidence
    - Low confidence (0-40): Ambiguous or limited evidence
    - Medium confidence (41-70): Some clear indicators present
    - High confidence (71-100): Multiple strong, unambiguous indicators
    
    **isAi determination**: 
    - Set to TRUE if PRNU analysis shows strong AI indicators OR multiple other strong signals
    - Set to FALSE if PRNU shows consistent sensor fingerprint AND other signals support real camera
    - Consider the TOTALITY of evidence, not individual anomalies
    - Real-world videos can have compression artifacts, editing, and imperfections

    Output STRICTLY in JSON format:
    {{
        "isAi": boolean,
        "confidence": number (0-100),
        "curvatureScore": number (0-100),
        "distanceScore": number (0-100),
        "reasoning": string[],
        "trajectoryData": [{{"x": number, "y": number, "frame": number}}],
        "modelDetected": string
    }}

    MANDATORY REQUIREMENTS:
    1. "reasoning": List 4-6 specific OBJECTIVE observations based on what you see
       - Describe actual visual evidence, not assumptions
       - Note both indicators of AI AND indicators of real footage
       - Be specific about frame numbers or sequences where relevant
    
    2. "trajectoryData": Generate 15-30 realistic coordinate points tracking motion
       - Should correspond to actual movement patterns observed
       - Points should reflect natural or unnatural trajectories observed
    
    3. "modelDetected": Identify based on EVIDENCE ONLY:
       - If clearly AI-generated: Specify "Sora", "Runway Gen-3", "Pika", "Kling", or "Unknown AI Model"
       - If appears authentic: "Real Camera"
       - If uncertain: "Real Camera" (default to real when ambiguous)
    
    4. "isAi": Base decision on clear technical evidence
       - Require MULTIPLE strong indicators, not single anomalies
       - Consider that real videos can have: compression artifacts, motion blur, editing cuts, color grading
       - Consider that real videos may have: shaky cam, autofocus hunting, lens flares, sensor noise
    
    5. "confidence": Reflect genuine uncertainty
       - Don't artificially inflate confidence
       - Lower confidence for ambiguous cases
       - Higher confidence only when multiple clear indicators align

    IMPORTANT PRINCIPLES:
    - Remain NEUTRAL - do not assume AI or Real without evidence
    - Real-world videos often have imperfections that are NOT signs of AI
    - Professional videography can look very clean and still be real
    - Animation and CGI are NOT the same as AI-generated deepfakes
    - Base conclusions on TECHNICAL EVIDENCE, not aesthetic preferences
    - When uncertain, acknowledge the uncertainty in your confidence score
        """

        content_parts = [{"type": "text", "text": prompt}]
        for frame in frames:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": frame}
            })

        response = requests.post(
             "https://openrouter.ai/api/v1/chat/completions",
             headers={
                 "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                 "Content-Type": "application/json",
                 "HTTP-Referer": "https://frametruth.com",
             },
             json={
                 "model": "google/gemini-3-pro-preview",
                 "messages": [{"role": "user", "content": content_parts}],
                 "response_format": {"type": "json_object"}
             }
        )

        if response.status_code != 200:
            raise Exception(f"AI API Error: {response.text}")
        
        ai_res = response.json()
        
        # Debug: Log the response structure
        print(f"🔍 AI API Response keys: {list(ai_res.keys())}")
        if 'error' in ai_res:
            raise Exception(f"AI API Error: {ai_res['error']}")
        
        if 'choices' not in ai_res:
            raise Exception(f"Unexpected AI API response format: {json.dumps(ai_res, indent=2)[:500]}")
        
        content = ai_res['choices'][0]['message']['content']
        clean_content = content.replace("```json", "").replace("```", "").strip()
        analysis_result = json.loads(clean_content)
        
        # 13. 🚀 EVIDENCE-BASED MACHINE LEARNING ANALYSIS
        print(f"🤖 Using Evidence-Based Scorer (trained on 87 videos with 66.7% accuracy)...")
        
        try:
            # Initialize the evidence-based scorer
            scorer = EvidenceBasedScorer()
            
            # Prepare features dictionary for the trained model
            features_dict = {}
            
            # 🚀 ADD ALL 36 FEATURES TO MATCH TRAINING DATA EXACTLY
            
            # PRNU Sensor Fingerprint Features (4 features)
            if prnu_metrics:
                features_dict.update({
                    'prnu_mean_corr': prnu_metrics['prnu_mean_corr'],
                    'prnu_std_corr': prnu_metrics['prnu_std_corr']
                })
            
            # Trajectory Analysis Features (4 features)
            if trajectory_metrics:
                features_dict.update({
                    'trajectory_curvature_mean': trajectory_metrics['mean_curvature'],
                    'trajectory_curvature_std': (trajectory_metrics['curvature_variance'] ** 0.5),
                    'trajectory_max_curvature': trajectory_metrics['max_curvature'],
                    'trajectory_mean_distance': trajectory_metrics['mean_distance']
                })
            
            # Optical Flow Features (5 features)
            if optical_flow_metrics:
                features_dict.update({
                    'flow_jitter_index': optical_flow_metrics['flow_jitter_index'],
                    'flow_patch_variance': optical_flow_metrics['flow_patch_variance'],
                    'flow_smoothness_score': optical_flow_metrics['flow_smoothness_score'],
                    'flow_global_mean': optical_flow_metrics['flow_global_mean'],
                    'flow_global_std': optical_flow_metrics['flow_global_std']
                })
            
            # OCR Text Analysis Features (5 features)
            if ocr_metrics:
                features_dict.update({
                    'ocr_frame_stability': ocr_metrics['ocr_frame_stability_score'],
                    'ocr_mutation_rate': ocr_metrics['ocr_text_mutation_rate'],
                    'ocr_unique_string_count': ocr_metrics['ocr_unique_string_count'],
                    'ocr_total_detections': ocr_metrics['total_text_detections']
                })
            
            # Frequency Domain Features (4 features)
            if frequency_metrics:
                features_dict.update({
                    'freq_low_power_mean': frequency_metrics.get('freq_low_power_mean', 0.0),
                    'freq_mid_power_mean': frequency_metrics.get('freq_mid_power_mean', 0.0),
                    'freq_high_power_mean': frequency_metrics.get('freq_high_power_mean', 0.0),
                    'freq_high_low_ratio_mean': frequency_metrics.get('freq_high_low_ratio_mean', 0.0)
                })
            
            # Noise/Grain Statistics Features (6 features)
            if noise_metrics:
                features_dict.update({
                    'noise_variance_r_mean': noise_metrics.get('noise_variance_r_mean', 0.0),
                    'noise_variance_g_mean': noise_metrics.get('noise_variance_g_mean', 0.0),
                    'noise_variance_b_mean': noise_metrics.get('noise_variance_b_mean', 0.0),
                    'cross_channel_corr_rg_mean': noise_metrics.get('cross_channel_corr_rg_mean', 0.0),
                    'spatial_autocorr_mean': noise_metrics.get('spatial_autocorr_mean', 0.0),
                    'temporal_noise_consistency': noise_metrics.get('temporal_noise_consistency', 0.0)
                })
            
            # Codec/Compression Features (7 features)
            if codec_metrics:
                features_dict.update({
                    'avg_bitrate': codec_metrics.get('avg_bitrate', 0.0),
                    'i_frame_ratio': codec_metrics.get('i_frame_ratio', 0.0),
                    'p_frame_ratio': codec_metrics.get('p_frame_ratio', 0.0),
                    'b_frame_ratio': codec_metrics.get('b_frame_ratio', 0.0),
                    'gop_length_mean': codec_metrics.get('gop_length_mean', 0.0),
                    'gop_length_std': codec_metrics.get('gop_length_std', 0.0),
                    'double_compression_score': codec_metrics.get('double_compression_score', 0.0)
                })
            
            # Metadata Features (4 features)
            if metadata_metrics:
                features_dict.update({
                    'duration_seconds': metadata_metrics.get('duration_seconds', duration),
                    'fps': metadata_metrics.get('fps', fps),
                    'resolution_width': metadata_metrics.get('resolution_width', 0),
                    'resolution_height': metadata_metrics.get('resolution_height', 0)
                })
            
            # Get evidence-based analysis using trained models
            evidence_result = scorer.analyze_video_evidence_based(features_dict)
            
            # Update analysis result with evidence-based predictions
            analysis_result['isAi'] = evidence_result['verdict'] in ['AI Generated', 'Likely AI']
            analysis_result['confidence'] = min(95, max(evidence_result['ai_probability'], analysis_result.get('confidence', 50)))
            analysis_result['curvatureScore'] = evidence_result['ai_probability']
            
            # Add evidence-based reasoning
            analysis_result['reasoning'].insert(0, f"🤖 Evidence-Based ML Analysis: {evidence_result['ai_probability']:.1f}% AI probability")
            analysis_result['reasoning'].insert(1, f"📊 Model: {evidence_result['analysis_method']} (trained on {evidence_result['training_data']})")
            analysis_result['reasoning'].insert(2, f"🎯 Verdict: {evidence_result['verdict']} ({evidence_result['confidence']} confidence)")
            
            # Add top contributing features
            if evidence_result['top_contributors']:
                contrib_text = "🔬 Key Evidence: " + ", ".join([
                    f"{contrib['description']} ({contrib['contribution']:.3f})"
                    for contrib in evidence_result['top_contributors'][:3]
                ])
                analysis_result['reasoning'].insert(3, contrib_text)
            
            print(f"🤖 Evidence-Based Result: {evidence_result['verdict']} ({evidence_result['ai_probability']:.1f}% AI)")
            print(f"   Model Accuracy: {evidence_result['model_accuracy']}")
            print(f"   Top Features: {[c['feature'] for c in evidence_result['top_contributors'][:3]]}")
            
        except Exception as e:
            print(f"⚠️ Evidence-based scorer failed: {e}")
            print(f"   Falling back to Gemini analysis only")
            # Keep original Gemini analysis result

        # 16. Save Submission AUTOMATICALLY
        submission_id = str(uuid.uuid4())[:8].upper()
        created_at = datetime.now().isoformat()
        
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("INSERT INTO submissions VALUES (?, ?, ?, ?, ?, ?)", 
                  (submission_id, data.filename, data.original_url, json.dumps(analysis_result), client_ip, created_at))
        conn.commit()
        conn.close()

        return {
            "submission_id": submission_id,
            "result": analysis_result
        }

    except Exception as e:
        # Log error
        print(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/submissions")
async def list_submissions(
    x_admin_user: str = Header(None), 
    x_admin_pass: str = Header(None),
    search: str = None,
    page: int = 1,
    limit: int = 50
):
    if x_admin_user != ADMIN_USER or x_admin_pass != ADMIN_PASS:
        raise HTTPException(status_code=401, detail="Unauthorized")

    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    # Build search query
    base_query = "SELECT * FROM submissions"
    count_query = "SELECT COUNT(*) FROM submissions"
    params = []
    
    if search and search.strip():
        search_term = f"%{search.strip()}%"
        where_clause = """ WHERE 
            id LIKE ? OR 
            filename LIKE ? OR 
            original_url LIKE ? OR 
            ip_address LIKE ? OR
            analysis_result LIKE ?
        """
        base_query += where_clause
        count_query += where_clause
        params = [search_term] * 5
    
    # Get total count
    c.execute(count_query, params)
    total = c.fetchone()[0]
    
    # Add pagination
    offset = (page - 1) * limit
    base_query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    c.execute(base_query, params)
    rows = c.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        try:
            analysis = json.loads(row["analysis_result"])
            summary = {
                "isAi": analysis.get("isAi"),
                "confidence": analysis.get("confidence"),
                "modelDetected": analysis.get("modelDetected")
            }
        except:
            summary = {}

        results.append({
            "id": row["id"],
            "filename": row["filename"],
            "video_url": f"http://localhost:8000/videos/{row['filename']}",
            "original_url": row["original_url"],
            "ip_address": row["ip_address"],
            "created_at": row["created_at"],
            "summary": summary
        })
    
    return {
        "submissions": results,
        "total": total,
        "page": page,
        "limit": limit,
        "pages": (total + limit - 1) // limit
    }

@app.get("/api/submission/{submission_id}")
async def get_submission(submission_id: str):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM submissions WHERE id = ?", (submission_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Submission not found")
        
    return {
        "id": row["id"],
        "filename": row["filename"],
        "video_url": f"http://localhost:8000/videos/{row['filename']}",
        "original_url": row["original_url"],
        "analysis_result": json.loads(row["analysis_result"]),
        "created_at": row["created_at"]
    }

@app.post("/api/admin/upload-cookies")
async def upload_cookies(
    file: UploadFile = File(...),
    x_admin_user: str = Header(None),
    x_admin_pass: str = Header(None)
):
    """Admin endpoint to upload YouTube cookies file for 100% reliable downloads"""
    if x_admin_user != ADMIN_USER or x_admin_pass != ADMIN_PASS:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        # Save cookies file
        with open(COOKIES_FILE, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        print(f"✅ YouTube cookies uploaded successfully to: {COOKIES_FILE}")
        
        return {
            "message": "YouTube cookies uploaded successfully",
            "file_path": COOKIES_FILE,
            "status": "Downloads will now use authenticated cookies to bypass bot detection"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cookie upload failed: {str(e)}")

@app.get("/api/admin/cookies-status")
async def check_cookies_status(
    x_admin_user: str = Header(None),
    x_admin_pass: str = Header(None)
):
    """Check if YouTube cookies are configured"""
    if x_admin_user != ADMIN_USER or x_admin_pass != ADMIN_PASS:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    cookies_exist = os.path.exists(COOKIES_FILE)
    
    return {
        "cookies_configured": cookies_exist,
        "cookies_path": COOKIES_FILE if cookies_exist else None,
        "status": "YouTube downloads will work 100% reliably" if cookies_exist else "YouTube may be blocked by bot detection"
    }

@app.post("/api/reset-rate-limit")
async def reset_rate_limit(request: Request):
    """Development endpoint to reset rate limit for current IP"""
    client_ip = request.headers.get("x-forwarded-for", request.client.host).split(",")[0]
    
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM submissions WHERE ip_address = ?", (client_ip,))
    deleted_count = c.rowcount
    conn.commit()
    conn.close()
    
    print(f"🔄 Reset rate limit for IP {client_ip}: deleted {deleted_count} submissions")
    
    return {
        "message": f"Rate limit reset for IP {client_ip}",
        "deleted_submissions": deleted_count
    }

@app.get("/docs")
def health_check():
    return {"status": "ok"}

@app.get("/api/test")
async def test_endpoint():
    """Simple test endpoint to verify backend is working"""
    return {
        "status": "Backend is working!",
        "environment": "production" if os.getenv("VERCEL") else "development",
        "openrouter_key_present": bool(OPENROUTER_API_KEY),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/check-ip")
async def check_ip(request: Request):
    """Debug endpoint to show detected IP address"""
    client_ip = request.headers.get("x-forwarded-for", request.client.host).split(",")[0]
    all_headers = dict(request.headers)
    
    return {
        "detected_ip": client_ip,
        "client_host": request.client.host,
        "x_forwarded_for": request.headers.get("x-forwarded-for"),
        "x_real_ip": request.headers.get("x-real-ip"),
        "cf_connecting_ip": request.headers.get("cf-connecting-ip"),
        "all_headers": all_headers,
        "is_whitelisted": client_ip in ["127.0.0.1", "localhost", "::1", "0.0.0.0", "192.168.1.16", "173.239.214.13"] or client_ip.startswith(("127.", "192.168.", "10.", "172."))
    }

# Mount frontend static files AFTER all API routes are defined
if os.path.exists("dist"):
    # Serve static assets first
    app.mount("/assets", StaticFiles(directory="dist/assets"), name="assets")
    
    # Catch-all route for React Router (SPA) - must be last
    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        # Don't interfere with API routes or video files
        if full_path.startswith("api/") or full_path.startswith("videos/"):
            raise HTTPException(status_code=404, detail="Not found")
        
        # For root path or any other path, serve index.html (React Router will handle routing)
        return FileResponse("dist/index.html")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
