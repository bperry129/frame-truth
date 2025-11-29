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
try:
    from enhanced_evidence_scorer import EnhancedEvidenceBasedScorer
    USE_ENHANCED_SCORER = True
    print("🚀 Using ENHANCED Evidence-Based Scorer (96.6% accuracy with optimized models)")
except ImportError:
    try:
        from improved_evidence_scorer import ImprovedEvidenceBasedScorer
        USE_ENHANCED_SCORER = False
        USE_IMPROVED_SCORER = True
        print("✅ Using IMPROVED Evidence-Based Scorer with manual corrections")
    except ImportError:
        from evidence_based_scorer import EvidenceBasedScorer
        USE_ENHANCED_SCORER = False
        USE_IMPROVED_SCORER = False
        print("⚠️ Using original Evidence-Based Scorer (66.7% accuracy)")

# Import advanced AI detector for sophisticated generators
try:
    from advanced_ai_detector import AdvancedAIDetector
    USE_ADVANCED_DETECTOR = True
    print("✅ Using ADVANCED AI Detector for sophisticated generators (Sora, Runway Gen-3)")
except ImportError:
    USE_ADVANCED_DETECTOR = False
    print("⚠️ Advanced AI Detector not available")

# Import feature extractors
sys.path.append(str(Path(__file__).parent.parent / "frametruth_training" / "feature_extractor"))
from frequency_features import compute_frequency_features
from noise_features import compute_noise_features
from codec_features import compute_codec_features
from metadata_features import compute_metadata_features
from trajectory_features import compute_trajectory_features

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
    🚀 FIXED PRNU ANALYSIS: Camera Sensor Fingerprint Consistency (PRNU)
    
    CRITICAL FIX: The previous implementation was giving FALSE POSITIVES for AI videos.
    AI videos were showing high correlations when they should show LOW correlations.
    
    Real cameras have a fixed, physical sensor with tiny imperfections:
    - Consistent noise pattern across frames (same sensor = same "grain fingerprint")
    - Stable PRNU signature that correlates strongly frame-to-frame
    
    AI videos:
    - Have SYNTHETIC noise that doesn't correlate between frames
    - Show RANDOM noise patterns without consistent sensor signature
    - Lack the physical sensor imperfections that create PRNU
    
    Returns:
    - prnu_mean_corr: Average correlation with global fingerprint (REAL: >0.3, AI: <0.15)
    - prnu_std_corr: Standard deviation of correlations (REAL: <0.2, AI: >0.3)
    - prnu_positive_ratio: Ratio of positive correlations (REAL: >0.7, AI: <0.5)
    - prnu_consistency_score: Overall consistency metric (REAL: >0.5, AI: <0.3)
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
        
        # FIXED APPROACH: More sophisticated PRNU extraction
        residuals = []
        for i, frame in enumerate(frames):
            # CRITICAL FIX: Use proper PRNU extraction method
            # Apply Gaussian blur to estimate the scene content
            blurred = cv2.GaussianBlur(frame, (5, 5), 1.0)
            
            # Calculate noise residual: original - scene estimate
            residual = frame - blurred
            
            # Apply high-pass filter to isolate sensor noise
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) / 8.0
            residual = cv2.filter2D(residual, -1, kernel)
            
            # Normalize residual to unit variance
            residual_std = np.std(residual)
            if residual_std > 1e-6:
                residual = residual / residual_std
            else:
                residual = residual * 0  # Zero out if no variation
            
            residuals.append(residual)
        
        # Step 2: Estimate global sensor fingerprint using median (more robust)
        # Stack residuals and take median to reduce random noise
        residuals_stack = np.stack(residuals, axis=2)
        global_fingerprint = np.median(residuals_stack, axis=2)
        
        # Normalize the global fingerprint
        fp_std = np.std(global_fingerprint)
        if fp_std > 1e-6:
            global_fingerprint = global_fingerprint / fp_std
        
        # Step 3: Measure per-frame correlation with global fingerprint
        correlations = []
        for residual in residuals:
            # Flatten for correlation calculation
            residual_flat = residual.flatten()
            fingerprint_flat = global_fingerprint.flatten()
            
            # Remove DC component (mean)
            residual_flat = residual_flat - np.mean(residual_flat)
            fingerprint_flat = fingerprint_flat - np.mean(fingerprint_flat)
            
            # Calculate Pearson correlation coefficient
            if np.std(residual_flat) > 1e-6 and np.std(fingerprint_flat) > 1e-6:
                correlation = np.corrcoef(residual_flat, fingerprint_flat)[0, 1]
            else:
                correlation = 0.0
            
            # Handle NaN case
            if np.isnan(correlation):
                correlation = 0.0
            
            correlations.append(correlation)
        
        # Step 4: Calculate PRNU metrics with CORRECTED INTERPRETATION
        correlations = np.array(correlations)
        
        prnu_mean_corr = float(np.mean(correlations))
        prnu_std_corr = float(np.std(correlations))
        
        # Count positive correlations (should be high for real cameras)
        positive_threshold = 0.05  # Lower threshold for more sensitivity
        prnu_positive_ratio = float(np.sum(correlations > positive_threshold) / len(correlations))
        
        # Calculate overall consistency score
        # FIXED: Real cameras should have high mean, low std, high positive ratio
        # AI videos should have low mean, high std, low positive ratio
        if prnu_std_corr > 0:
            consistency_score = (prnu_mean_corr * prnu_positive_ratio) / (prnu_std_corr + 0.01)
        else:
            consistency_score = prnu_mean_corr * prnu_positive_ratio
        
        prnu_consistency_score = float(max(0, min(1, consistency_score)))
        
        # DEBUGGING: Print actual values to understand what's happening
        print(f"   🔬 PRNU Debug: mean_corr={prnu_mean_corr:.3f}, std_corr={prnu_std_corr:.3f}, pos_ratio={prnu_positive_ratio:.3f}")
        
        return {
            'prnu_mean_corr': prnu_mean_corr,
            'prnu_std_corr': prnu_std_corr,
            'prnu_positive_ratio': prnu_positive_ratio,
            'prnu_consistency_score': prnu_consistency_score,
            'num_samples': len(frames),
            'method': 'prnu_sensor_fingerprint_fixed'
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

def extract_frames_numpy(video_path, num_frames=15):
    """Extract frames as numpy arrays for ReStraV analysis"""
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
            frames.append(frame)
            extracted += 1
            
        count += 1

    cap.release()
    return frames

async def run_hybrid_analysis(video_path: str, original_url: str = "") -> dict:
    """
    Run hybrid Gemini + ReStraV analysis in parallel
    
    Returns combined analysis with weighted scoring
    """
    print(f"🔬 Starting hybrid Gemini + ReStraV analysis...")
    
    try:
        # Extract frames for both analyses
        print(f"🖼️ Extracting frames for dual analysis...")
        
        # For Gemini: Base64 encoded frames
        gemini_frames = extract_frames_base64(video_path, 20)
        
        # For ReStraV: Numpy arrays
        restrav_frames = extract_frames_numpy(video_path, 15)
        
        if not gemini_frames or not restrav_frames:
            raise Exception("Could not extract frames for analysis")
        
        # Run Gemini analysis
        print(f"🤖 Running Gemini visual analysis...")
        gemini_result = await run_gemini_analysis(gemini_frames)
        
        # Run ReStraV analysis
        print(f"📊 Running ReStraV trajectory analysis...")
        restrav_result = run_restrav_analysis(restrav_frames, video_path)
        
        # Run frequency domain analysis
        print(f"🔊 Running frequency domain analysis...")
        frequency_result = run_frequency_analysis(video_path)
        
        # Combine results with weighted scoring
        print(f"⚖️ Combining results with weighted scoring...")
        hybrid_result = combine_analysis_results(gemini_result, restrav_result, original_url, frequency_result)
        
        return hybrid_result
        
    except Exception as e:
        print(f"❌ Hybrid analysis failed: {str(e)}")
        raise

async def run_gemini_analysis(frames: list) -> dict:
    """Run Gemini visual analysis on frames"""
    try:
        prompt = """You are an expert AI video detection specialist with deep knowledge of modern AI video generators (Sora, Runway Gen-3, Pika, Kling, etc.). Analyze these video frames with extreme scrutiny to determine if this video was generated by AI or captured with a real camera.

**CRITICAL AI DETECTION SIGNALS - Examine Carefully:**

**Micro-Level AI Artifacts:**
- **Temporal flickering**: Subtle brightness/color changes between frames in static areas
- **Edge inconsistencies**: Object edges that slightly shift or "breathe" between frames
- **Texture boiling**: Fine textures (hair, fabric, skin) that subtly change pattern between frames
- **Lighting micro-inconsistencies**: Shadows or highlights that don't maintain perfect consistency
- **Compression anomalies**: Unusual compression patterns that differ from real camera footage
- **Pixel-level artifacts**: Unnatural pixel patterns, especially in gradients and smooth areas

**Advanced AI Signatures:**
- **Motion uncanny valley**: Movement that's almost natural but feels slightly "off"
- **Physics violations**: Subtle violations of physics (gravity, momentum, collision)
- **Temporal coherence issues**: Objects maintaining impossible consistency across frames
- **Depth perception errors**: Foreground/background relationships that don't make physical sense
- **Reflection/shadow mismatches**: Reflections or shadows that don't perfectly match their sources

**Generator-Specific Patterns:**
- **Sora**: Overly smooth motion, perfect temporal consistency, unnatural camera movements
- **Runway Gen-3**: Slight texture warping, motion blur inconsistencies
- **Pika/Kling**: Frame-to-frame jitter, temporal artifacts in fine details

**Real Camera Indicators:**
- **Authentic sensor noise**: Consistent noise patterns across frames
- **Natural compression**: Realistic compression artifacts from actual camera codecs
- **Physical camera limitations**: Natural motion blur, focus hunting, exposure adjustments
- **Authentic imperfections**: Real-world lighting variations, natural camera shake

**ANALYSIS INSTRUCTIONS:**
1. Compare IDENTICAL areas across multiple frames - look for micro-changes
2. Examine fine details (hair, fabric texture, skin pores) for temporal consistency
3. Check if motion follows realistic physics and camera behavior
4. Look for any "too perfect" elements that real cameras wouldn't capture
5. Pay special attention to areas that should be static but show subtle changes

Provide your analysis in this exact JSON format:
{
    "isAi": boolean,
    "confidence": number (0-100),
    "curvatureScore": number (0-100),
    "distanceScore": number (0-100),
    "reasoning": ["observation 1", "observation 2", "observation 3", "observation 4"],
    "trajectoryData": [{"x": number, "y": number, "frame": number}],
    "modelDetected": "Real Camera" or "Sora" or "Runway Gen-3" or "Pika" or "Kling" or "Unknown AI Model"
}

Be extremely thorough and suspicious. Modern AI videos can be very convincing - look for the subtle tells."""

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
                 "model": "google/gemini-2.0-flash-001",
                 "messages": [{"role": "user", "content": content_parts}],
                 "response_format": {"type": "json_object"}
             }
        )

        if response.status_code != 200:
            raise Exception(f"Gemini API Error: {response.text}")
        
        ai_res = response.json()
        
        if 'error' in ai_res:
            raise Exception(f"Gemini API Error: {ai_res['error']}")
        
        if 'choices' not in ai_res:
            raise Exception(f"Unexpected Gemini API response format")
        
        content = ai_res['choices'][0]['message']['content']
        clean_content = content.replace("```json", "").replace("```", "").strip()
        gemini_result = json.loads(clean_content)
        
        print(f"✅ Gemini analysis complete: {gemini_result.get('confidence', 0)}% confidence")
        
        return gemini_result
        
    except Exception as e:
        print(f"❌ Gemini analysis failed: {str(e)}")
        # Return default result to allow ReStraV to still work
        return {
            "isAi": False,
            "confidence": 0,
            "curvatureScore": 0,
            "distanceScore": 0,
            "reasoning": [f"Gemini analysis failed: {str(e)}"],
            "trajectoryData": [],
            "modelDetected": "Analysis Error"
        }

def run_restrav_analysis(frames: list, video_path: str) -> dict:
    """Run ReStraV trajectory analysis on frames"""
    try:
        # Try to get DINOv2 model for enhanced analysis
        try:
            dinov2_model, dinov2_transform = get_dinov2_model()
            print(f"✅ Using DINOv2 for enhanced ReStraV analysis")
        except Exception as e:
            print(f"⚠️ DINOv2 not available, using lightweight features: {e}")
            dinov2_model, dinov2_transform = None, None
        
        # Run trajectory analysis
        trajectory_result = compute_trajectory_features(
            frames, 
            dinov2_model=dinov2_model, 
            dinov2_transform=dinov2_transform
        )
        
        print(f"✅ ReStraV analysis complete: {trajectory_result.get('restrav_confidence', 0)}% AI confidence")
        
        return trajectory_result
        
    except Exception as e:
        print(f"❌ ReStraV analysis failed: {str(e)}")
        # Return default result
        return {
            "restrav_confidence": 0,
            "trajectory_curvature_mean": 0,
            "trajectory_curvature_variance": 0,
            "trajectory_smoothness": 0,
            "trajectory_consistency": 0,
            "method": "error"
        }

def run_frequency_analysis(video_path: str) -> dict:
    """
    Run frequency domain analysis to detect AI artifacts
    
    AI videos often have different frequency signatures than real camera footage:
    - Unnatural DCT coefficient distributions
    - Missing high-frequency noise patterns
    - Artificial compression artifacts
    """
    try:
        print(f"🔊 Running frequency domain analysis...")
        
        # Extract frames for frequency analysis
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            return {"frequency_confidence": 0, "method": "error"}
        
        # Sample 8 frames for frequency analysis (balance speed vs accuracy)
        step = max(1, total_frames // 8)
        frames = []
        count = 0
        extracted = 0
        
        while cap.isOpened() and extracted < 8:
            ret, frame = cap.read()
            if not ret:
                break
                
            if count % step == 0:
                # Convert to grayscale and resize for consistent analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_resized = cv2.resize(gray, (256, 256))
                frames.append(gray_resized)
                extracted += 1
                    
            count += 1
        
        cap.release()
        
        if len(frames) < 2:
            return {"frequency_confidence": 0, "method": "insufficient_frames"}
        
        # Use the existing frequency features function
        frequency_features = compute_frequency_features(frames)
        
        if not frequency_features:
            return {"frequency_confidence": 0, "method": "feature_extraction_failed"}
        
        # Calculate AI confidence based on frequency features
        ai_indicators = 0
        total_indicators = 0
        
        # DCT energy distribution (AI videos often have unnatural distributions)
        dct_energy_mean = frequency_features.get('dct_energy_mean', 0)
        dct_energy_std = frequency_features.get('dct_energy_std', 0)
        
        # Real videos: moderate energy with variation
        # AI videos: either too uniform or too chaotic
        if dct_energy_std < 0.1 or dct_energy_std > 2.0:
            ai_indicators += 1
        total_indicators += 1
        
        # High frequency content (AI videos often lack natural high-freq noise)
        high_freq_ratio = frequency_features.get('high_freq_ratio', 0.5)
        if high_freq_ratio < 0.15:  # Too little high frequency content
            ai_indicators += 1
        total_indicators += 1
        
        # Spectral rolloff (frequency distribution shape)
        spectral_rolloff = frequency_features.get('spectral_rolloff', 0.5)
        if spectral_rolloff < 0.3 or spectral_rolloff > 0.8:  # Unnatural distribution
            ai_indicators += 1
        total_indicators += 1
        
        # Frequency variance across frames (temporal consistency)
        freq_variance = frequency_features.get('frequency_variance', 0)
        if freq_variance < 0.01:  # Too consistent (AI smoothing)
            ai_indicators += 1
        total_indicators += 1
        
        # Calculate confidence as percentage of AI indicators
        frequency_confidence = (ai_indicators / max(total_indicators, 1)) * 100
        
        print(f"✅ Frequency analysis complete: {frequency_confidence:.1f}% AI confidence")
        print(f"   📊 AI indicators: {ai_indicators}/{total_indicators}")
        
        return {
            "frequency_confidence": frequency_confidence,
            "ai_indicators": ai_indicators,
            "total_indicators": total_indicators,
            "dct_energy_mean": dct_energy_mean,
            "dct_energy_std": dct_energy_std,
            "high_freq_ratio": high_freq_ratio,
            "spectral_rolloff": spectral_rolloff,
            "frequency_variance": freq_variance,
            "method": "dct_spectral_analysis"
        }
        
    except Exception as e:
        print(f"❌ Frequency analysis failed: {str(e)}")
        return {
            "frequency_confidence": 0,
            "method": "error",
            "error": str(e)
        }

def combine_analysis_results(gemini_result: dict, restrav_result: dict, original_url: str = "", frequency_result: dict = None) -> dict:
    """
    Combine Gemini, ReStraV, and frequency analysis results with intelligent weighting
    """
    try:
        # Extract scores
        gemini_confidence = gemini_result.get('confidence', 0)
        gemini_is_ai = gemini_result.get('isAi', False)
        
        restrav_confidence = restrav_result.get('restrav_confidence', 0)
        restrav_method = restrav_result.get('method', 'unknown')
        
        # Extract frequency analysis results
        frequency_confidence = 0
        frequency_method = "not_available"
        if frequency_result:
            frequency_confidence = frequency_result.get('frequency_confidence', 0)
            frequency_method = frequency_result.get('method', 'unknown')
        
        # Determine video characteristics for dynamic weighting
        video_characteristics = analyze_video_characteristics(gemini_result, restrav_result, original_url)
        
        # Calculate dynamic weights based on video characteristics
        weights = calculate_dynamic_weights(video_characteristics)
        gemini_weight = weights['gemini']
        restrav_weight = weights['restrav']
        
        print(f"📊 Dynamic weights: Gemini={gemini_weight:.2f}, ReStraV={restrav_weight:.2f}")
        print(f"📊 Video characteristics: {video_characteristics}")
        
        # Combine confidence scores
        if gemini_is_ai:
            # If Gemini says AI, use Gemini confidence directly
            gemini_ai_score = gemini_confidence
        else:
            # If Gemini says real, invert the confidence
            gemini_ai_score = 100 - gemini_confidence
        
        # ReStraV confidence is already AI confidence (0-100)
        restrav_ai_score = restrav_confidence
        
        # Weighted combination
        combined_ai_score = (gemini_ai_score * gemini_weight) + (restrav_ai_score * restrav_weight)
        combined_ai_score = max(0, min(100, combined_ai_score))  # Clamp to 0-100
        
        # Determine final classification
        final_is_ai = combined_ai_score > 50
        final_confidence = combined_ai_score if final_is_ai else (100 - combined_ai_score)
        
        # Enhanced reasoning that combines both analyses
        combined_reasoning = []
        
        # Add Gemini reasoning
        gemini_reasoning = gemini_result.get('reasoning', [])
        for reason in gemini_reasoning[:2]:  # Take top 2 Gemini reasons
            combined_reasoning.append(f"Visual: {reason}")
        
        # Add ReStraV reasoning
        if restrav_confidence > 20:
            if restrav_result.get('trajectory_curvature_variance', 0) > 400:
                combined_reasoning.append("Motion: Irregular trajectory patterns suggest AI generation")
            if restrav_result.get('trajectory_smoothness', 1) < 0.01:
                combined_reasoning.append("Motion: Unnatural motion smoothness detected")
        elif restrav_confidence < 10:
            combined_reasoning.append("Motion: Consistent natural motion patterns")
        
        # Add frequency analysis reasoning
        if frequency_confidence > 50:
            combined_reasoning.append("Frequency: Unnatural frequency domain patterns detected")
        elif frequency_confidence > 25:
            combined_reasoning.append("Frequency: Some artificial frequency signatures found")
        elif frequency_method == "dct_spectral_analysis":
            combined_reasoning.append("Frequency: Natural frequency patterns consistent with real camera")
        
        # Add method information
        analysis_methods = []
        if restrav_method == "dinov2_restrav":
            analysis_methods.append("Enhanced ReStraV with DINOv2")
        elif restrav_method == "lightweight_restrav":
            analysis_methods.append("Lightweight ReStraV")
        analysis_methods.append("Gemini 2.0 Flash")
        if frequency_method == "dct_spectral_analysis":
            analysis_methods.append("Frequency Analysis")
        
        combined_reasoning.append(f"Analysis: {' + '.join(analysis_methods)}")
        
        # Determine model detected with hybrid logic
        model_detected = determine_hybrid_model(gemini_result, restrav_result, final_is_ai)
        
        # Create enhanced trajectory data (combine both if available)
        trajectory_data = gemini_result.get('trajectoryData', [])
        
        # Build final result
        hybrid_result = {
            "isAi": final_is_ai,
            "confidence": round(final_confidence, 1),
            "curvatureScore": gemini_result.get('curvatureScore', 0),
            "distanceScore": gemini_result.get('distanceScore', 0),
            "reasoning": combined_reasoning,
            "trajectoryData": trajectory_data,
            "modelDetected": model_detected,
            
            # Hybrid-specific fields
            "analysisMethod": "hybrid_gemini_restrav",
            "geminiScore": round(gemini_ai_score, 1),
            "restravScore": round(restrav_ai_score, 1),
            "combinedScore": round(combined_ai_score, 1),
            "weights": {
                "gemini": round(gemini_weight, 2),
                "restrav": round(restrav_weight, 2)
            },
            "videoCharacteristics": video_characteristics,
            
            # Detailed analysis breakdown
            "detailedAnalysis": {
                "gemini": {
                    "confidence": gemini_confidence,
                    "isAi": gemini_is_ai,
                    "reasoning": gemini_reasoning
                },
                "restrav": {
                    "confidence": restrav_confidence,
                    "method": restrav_method,
                    "curvature_variance": restrav_result.get('trajectory_curvature_variance', 0),
                    "smoothness": restrav_result.get('trajectory_smoothness', 0),
                    "consistency": restrav_result.get('trajectory_consistency', 0)
                },
                "frequency": {
                    "confidence": frequency_confidence,
                    "method": frequency_method,
                    "ai_indicators": frequency_result.get('ai_indicators', 0) if frequency_result else 0,
                    "total_indicators": frequency_result.get('total_indicators', 0) if frequency_result else 0,
                    "high_freq_ratio": frequency_result.get('high_freq_ratio', 0) if frequency_result else 0,
                    "spectral_rolloff": frequency_result.get('spectral_rolloff', 0) if frequency_result else 0
                }
            }
        }
        
        print(f"🎯 Hybrid result: {model_detected} ({final_confidence}% confidence)")
        
        return hybrid_result
        
    except Exception as e:
        print(f"❌ Result combination failed: {str(e)}")
        # Fallback to Gemini result if combination fails
        return gemini_result

def analyze_video_characteristics(gemini_result: dict, restrav_result: dict, original_url: str = "") -> dict:
    """Analyze video characteristics to determine optimal weighting strategy"""
    characteristics = {
        "has_text": False,
        "has_motion": False,
        "has_faces": False,
        "is_static": False,
        "platform": "unknown"
    }
    
    # Analyze Gemini reasoning for content characteristics
    gemini_reasoning = " ".join(gemini_result.get('reasoning', [])).lower()
    
    if any(word in gemini_reasoning for word in ['text', 'writing', 'letters', 'words']):
        characteristics["has_text"] = True
    
    if any(word in gemini_reasoning for word in ['face', 'facial', 'person', 'human']):
        characteristics["has_faces"] = True
    
    if any(word in gemini_reasoning for word in ['static', 'still', 'no motion', 'stationary']):
        characteristics["is_static"] = True
    
    # Analyze ReStraV results for motion characteristics
    curvature_variance = restrav_result.get('trajectory_curvature_variance', 0)
    distance_mean = restrav_result.get('trajectory_distance_mean', 0)
    
    if curvature_variance > 100 or distance_mean > 0.1:
        characteristics["has_motion"] = True
    
    # Determine platform from URL
    if original_url:
        if "youtube.com" in original_url or "youtu.be" in original_url:
            characteristics["platform"] = "youtube"
        elif "tiktok.com" in original_url:
            characteristics["platform"] = "tiktok"
        elif "instagram.com" in original_url:
            characteristics["platform"] = "instagram"
    
    return characteristics

def calculate_dynamic_weights(characteristics: dict) -> dict:
    """Calculate dynamic weights based on video characteristics"""
    # Base weights (balanced)
    gemini_weight = 0.65  # Gemini is generally strong at visual artifacts
    restrav_weight = 0.35  # ReStraV is strong at motion patterns
    
    # Adjust based on characteristics
    if characteristics.get("has_text", False):
        # Text-heavy videos: Gemini is much better at text analysis
        gemini_weight += 0.15
        restrav_weight -= 0.15
    
    if characteristics.get("has_motion", False) and not characteristics.get("is_static", False):
        # High-motion videos: ReStraV is more valuable
        restrav_weight += 0.10
        gemini_weight -= 0.10
    
    if characteristics.get("is_static", False):
        # Static videos: Gemini is more valuable (ReStraV less useful)
        gemini_weight += 0.20
        restrav_weight -= 0.20
    
    if characteristics.get("platform") == "tiktok":
        # TikTok videos often have rapid motion: favor ReStraV slightly
        restrav_weight += 0.05
        gemini_weight -= 0.05
    
    # Ensure weights sum to 1.0 and are within reasonable bounds
    total_weight = gemini_weight + restrav_weight
    gemini_weight = max(0.4, min(0.8, gemini_weight / total_weight))
    restrav_weight = 1.0 - gemini_weight
    
    return {
        "gemini": gemini_weight,
        "restrav": restrav_weight
    }

def determine_hybrid_model(gemini_result: dict, restrav_result: dict, is_ai: bool) -> str:
    """Determine the most likely model based on hybrid analysis"""
    if not is_ai:
        return "Real Camera"
    
    # Start with Gemini's model detection
    gemini_model = gemini_result.get('modelDetected', 'Unknown AI Model')
    
    # Enhance with ReStraV characteristics
    curvature_variance = restrav_result.get('trajectory_curvature_variance', 0)
    smoothness = restrav_result.get('trajectory_smoothness', 0)
    
    # High curvature variance + low smoothness often indicates Sora
    if curvature_variance > 800 and smoothness < 0.001:
        if "sora" not in gemini_model.lower():
            return "Sora (motion analysis)"
    
    # Very high smoothness might indicate Runway Gen-3
    elif smoothness > 100 and curvature_variance < 200:
        if "runway" not in gemini_model.lower():
            return "Runway Gen-3 (motion analysis)"
    
    # Default to Gemini's detection
    return gemini_model

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
            # Strategy 1: Direct download with proper YouTube headers and range support
            {
                'name': 'YouTube Browser with Range',
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Referer': 'https://www.youtube.com/',
                    'Accept': '*/*',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'identity',  # Don't use compression for video
                    'Connection': 'keep-alive',
                    'Range': 'bytes=0-'  # Request from beginning
                },
                'timeout': 60,
                'stream': True
            },
            # Strategy 2: Mobile browser with range support
            {
                'name': 'Mobile Browser with Range',
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1',
                    'Accept': '*/*',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'identity',
                    'Connection': 'keep-alive',
                    'Range': 'bytes=0-'
                },
                'timeout': 60,
                'stream': True
            },
            # Strategy 3: Direct download without range (sometimes works)
            {
                'name': 'Direct Download',
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': '*/*'
                },
                'timeout': 45,
                'stream': True
            },
            # Strategy 4: Minimal approach
            {
                'name': 'Minimal Headers',
                'headers': {
                    'User-Agent': 'Mozilla/5.0'
                },
                'timeout': 30,
                'stream': True
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

async def download_with_alternative_youtube_apis(url: str, file_id: str) -> dict:
    """
    REMOVED: Alternative YouTube downloaders were unreliable
    This function now just raises an exception to trigger the user guidance
    """
    print(f"🔄 Skipping unreliable alternative YouTube APIs...")
    raise Exception("Alternative YouTube APIs disabled - they were unreliable")

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
        
        # SIMPLIFIED: Skip complex YouTube detection - go straight to fallbacks
        if is_youtube:
            print(f"🎥 YouTube detected - using simplified approach with fast fallback")
            # Minimal yt-dlp attempt with immediate fallback
            ydl_opts.update({
                'socket_timeout': 10,  # Very short timeout for fast fallback
                'retries': 2,  # Minimal retries
                'fragment_retries': 2,
                'file_access_retries': 2,
                'format': 'worst[height<=480]/worst',  # Low quality for speed
                'quiet': True,  # Reduce noise
                'no_warnings': True,
            })
        else:
            print(f"📱 Non-YouTube platform detected - using standard settings")
        
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
            
            # CRITICAL: Try cookie-based download for YouTube if available
            if is_youtube and os.path.exists(COOKIES_FILE):
                print(f"🔄 Attempting cookie-based download (final fallback)...")
                try:
                    result = await download_with_cookies(request.url, file_id)
                    print(f"✅ Cookie-based download successful!")
                    return result
                except Exception as cookie_error:
                    print(f"❌ Cookie-based download failed: {str(cookie_error)}")
                    
                    # Check if cookies are expired and provide helpful message
                    if "no longer valid" in str(cookie_error) or "rotated" in str(cookie_error):
                        print(f"🍪 COOKIES EXPIRED: YouTube cookies need to be refreshed")
            
            # FINAL FALLBACK: Try alternative YouTube downloaders
            print(f"🔄 Attempting alternative YouTube downloaders (last resort)...")
            try:
                result = await download_with_alternative_youtube_apis(request.url, file_id)
                print(f"✅ Alternative YouTube API successful!")
                return result
            except Exception as alt_error:
                print(f"❌ Alternative YouTube APIs failed: {str(alt_error)}")
            
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
        # 3. Calculate video duration for logging
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 30
        cap.release()
        
        print(f"📹 Video ({duration:.1f}s) - starting hybrid Gemini + ReStraV analysis")
        
        # 4. Run Hybrid Analysis (Gemini + ReStraV)
        analysis_result = await run_hybrid_analysis(filepath, data.original_url)
        
        print(f"🎯 Hybrid Analysis Complete: {analysis_result.get('modelDetected', 'Unknown')} ({analysis_result.get('confidence', 0)}% confidence)")

        # 5. Save Submission AUTOMATICALLY
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
        print(f"Hybrid analysis failed: {e}")
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
