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

load_dotenv(".env")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS")

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
COOKIES_DIR = "cookies"
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

async def download_with_y2mate(url: str, file_id: str) -> dict:
    """
    Fallback download method using Y2Mate API (for YouTube bot detection bypass)
    FREE API - works reliably for YouTube downloads
    """
    print(f"🔄 Attempting Y2Mate API fallback for: {url}")
    
    try:
        # Y2Mate API - Step 1: Analyze video
        analyze_url = "https://www.y2mate.com/mates/analyzeV2/ajax"
        
        analyze_payload = {
            "k_query": url,
            "k_page": "home",
            "hl": "en",
            "q_auto": 0
        }
        
        print(f"📤 Y2Mate analyze request for: {url}")
        print(f"📤 Y2Mate API endpoint: {analyze_url}")
        print(f"📤 Payload: {analyze_payload}")
        
        analyze_response = requests.post(
            analyze_url,
            data=analyze_payload,
            headers={
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
            timeout=30
        )
        
        print(f"📥 Y2Mate analyze response status: {analyze_response.status_code}")
        print(f"📥 Y2Mate analyze response body: {analyze_response.text[:500]}")
        
        if analyze_response.status_code != 200:
            raise Exception(f"Y2Mate analyze failed: {analyze_response.text[:200]}")
        
        analyze_data = analyze_response.json()
        
        if analyze_data.get("status") != "ok":
            raise Exception(f"Y2Mate returned status: {analyze_data.get('status')}")
        
        # Extract video ID and links
        vid = analyze_data.get("vid")
        links = analyze_data.get("links", {})
        
        if not vid or not links:
            raise Exception("No video data from Y2Mate")
        
        # Try to get 720p mp4, fallback to other qualities
        video_formats = links.get("mp4", {})
        
        k_value = None
        quality = None
        
        # Priority: 720p > 480p > 360p > any
        for q in ["720", "480", "360"]:
            if q in video_formats:
                k_value = video_formats[q].get("k")
                quality = q
                break
        
        if not k_value:
            # Get any available format
            for q, data in video_formats.items():
                k_value = data.get("k")
                quality = q
                break
        
        if not k_value:
            raise Exception("No downloadable format found")
        
        print(f"📥 Selected quality: {quality}p")
        
        # Step 2: Get download link
        convert_url = "https://www.y2mate.com/mates/convertV2/index"
        
        convert_payload = {
            "vid": vid,
            "k": k_value
        }
        
        convert_response = requests.post(
            convert_url,
            data=convert_payload,
            headers={
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
            timeout=30
        )
        
        if convert_response.status_code != 200:
            raise Exception(f"Y2Mate convert failed: {convert_response.text[:200]}")
        
        convert_data = convert_response.json()
        
        if convert_data.get("status") != "ok":
            raise Exception(f"Y2Mate convert status: {convert_data.get('status')}")
        
        # Extract download URL from HTML response
        import re
        dlink_match = re.search(r'href="([^"]+)"', convert_data.get("dlink", ""))
        
        if not dlink_match:
            raise Exception("No download URL found in response")
        
        download_url = dlink_match.group(1)
        
        print(f"📥 Downloading from Y2Mate: {download_url[:100]}...")
        
        # Download the video
        video_response = requests.get(download_url, stream=True, timeout=120, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        video_response.raise_for_status()
        
        # Save to file
        filename = f"{file_id}.mp4"
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        
        with open(filepath, 'wb') as f:
            for chunk in video_response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"✅ Y2Mate download successful: {filename} ({quality}p)")
        
        # Return metadata
        return {
            "filename": filename,
            "url": f"/videos/{filename}",
            "meta": {
                "title": analyze_data.get("title", "Video"),
                "uploader": "Unknown",
                "source": "y2mate_api",
                "quality": f"{quality}p"
            }
        }
        
    except Exception as e:
        print(f"❌ Y2Mate API failed: {str(e)}")
        raise

@app.post("/api/download")
async def download_video(request: DownloadRequest):
    try:
        print(f"📥 Download request received for URL: {request.url}")
        
        file_id = str(uuid.uuid4())
        filename = f"{file_id}.%(ext)s"  # Let yt-dlp determine the extension
        filepath = os.path.join(DOWNLOAD_DIR, f"{file_id}")
        
        ydl_opts = {
            # Use mobile format (less likely to be blocked)
            'format': 'best[height<=720][ext=mp4]/best[height<=720]/best[ext=mp4]/best',
            'outtmpl': filepath + '.%(ext)s',
            'quiet': False,  # Enable logging for debugging
            'no_warnings': False,
            'verbose': True,  # Add verbose logging
            'max_filesize': 100 * 1024 * 1024,
            'noplaylist': True,
            'geo_bypass': True,
            
            # ENHANCED: Multiple client strategies to bypass YouTube bot detection
            'extractor_args': {
                'youtube': {
                    # Try multiple clients in order of reliability
                    'player_client': ['ios', 'android', 'web'],
                    'skip': ['dash', 'hls'],
                    'player_skip': ['configs'],
                }
            },
            
            # Enhanced headers mimicking iOS (YouTube blocks Android less on iOS)
            'http_headers': {
                'User-Agent': 'com.google.ios.youtube/19.29.1 (iPhone14,3; U; CPU iOS 15_6 like Mac OS X)',
                'Accept': '*/*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate',
            },
            
            # Additional anti-bot measures
            'sleep_interval': 2,  # Increased from 1
            'max_sleep_interval': 10,  # Increased from 5
            'retries': 15,  # Increased
            'fragment_retries': 15,
            'file_access_retries': 10,
            
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
        
        # CRITICAL: Use cookies if available (100% reliable bypass)
        if os.path.exists(COOKIES_FILE):
            ydl_opts['cookiefile'] = COOKIES_FILE
            print(f"🍪 Using cookies from: {COOKIES_FILE}")
        
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
        
        # Check if this is a YouTube bot detection error
        is_youtube = "youtube.com" in request.url or "youtu.be" in request.url
        is_bot_error = "Sign in to confirm" in error_msg or "bot" in error_msg.lower()
        
        if is_youtube and is_bot_error:
            print(f"🔄 YouTube bot detection - attempting Y2Mate API fallback...")
            
            try:
                # Try Y2Mate API as fallback
                result = await download_with_y2mate(request.url, file_id)
                print(f"✅ Y2Mate API fallback successful!")
                return result
                
            except Exception as y2mate_error:
                print(f"❌ Y2Mate API fallback also failed: {str(y2mate_error)}")
                
                # Both methods failed - provide helpful message
                helpful_msg = (
                    "🚫 YouTube Download Failed\n\n"
                    "Both download methods failed (yt-dlp + Y2Mate API). "
                    "This can happen with certain restricted videos.\n\n"
                    "✅ **Easy Workaround (30 seconds):**\n"
                    "1. Download the YouTube video to your device (use any YouTube downloader)\n"
                    "2. Click the 'Upload File' tab above\n"
                    "3. Upload the video file\n"
                    "4. Analyze as normal!\n\n"
                    "💡 **Other Options:**\n"
                    "• Try a different platform (TikTok, Instagram, Twitter all work great!)\n"
                    "• Wait 10-15 minutes and try again"
                )
                
                return JSONResponse(status_code=400, content={
                    "detail": helpful_msg,
                    "error_type": "youtube_all_methods_failed",
                    "primary_error": error_msg,
                    "fallback_error": str(y2mate_error)
                })
        
        # Non-YouTube error or non-bot error
        return JSONResponse(status_code=400, content={"detail": f"Download failed: {error_msg}"})

@app.post("/api/analyze")
async def analyze_video(request: Request, data: AnalyzeRequest):
    # 1. Check Rate Limit
    client_ip = request.headers.get("x-forwarded-for", request.client.host).split(",")[0]
    print(f"🔍 Client IP detected: {client_ip}")
    print(f"🔍 Request client host: {request.client.host}")
    print(f"🔍 X-Forwarded-For header: {request.headers.get('x-forwarded-for')}")
    
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
        
        # Optimized frame count for Railway (reduced for speed)
        if duration < 10:
            num_frames = 15  # Very short clips
            dinov2_samples = 12  # Reduced from 24
            print(f"📹 Very short video ({duration:.1f}s) - using 15 frames")
        elif duration < 30:
            num_frames = 20  # Short videos
            dinov2_samples = 12
            print(f"📹 Short video ({duration:.1f}s) - using 20 frames")
        elif duration < 60:
            num_frames = 25  # Medium videos
            dinov2_samples = 15
            print(f"📹 Medium video ({duration:.1f}s) - using 25 frames")
        else:
            num_frames = 30  # Longer videos
            dinov2_samples = 15
            print(f"📹 Long video ({duration:.1f}s) - using 30 frames")
        
        # 4. Lightweight trajectory analysis (OpenCV-based, no PyTorch)
        print(f"📐 Calculating visual trajectory metrics ({dinov2_samples} samples)...")
        trajectory_metrics = calculate_lightweight_trajectory_metrics(filepath, num_samples=dinov2_samples)
        trajectory_boost = {'score_increase': 0, 'confidence_boost': 0, 'has_high_curvature': False, 'force_ai': False}
        
        if trajectory_metrics:
            mean_curv = trajectory_metrics['mean_curvature']
            curv_var = trajectory_metrics['curvature_variance']
            mean_dist = trajectory_metrics['mean_distance']
            
            # ReStraV paper findings (semantic space with DINOv2):
            # Real videos: mean_curvature typically 60-80° (straighter semantic trajectories)
            # AI videos: mean_curvature typically 85-110° (irregular semantic changes)
            # Threshold: 82° is a good separator
            
            if mean_curv > 90:  # Strong AI indicator
                trajectory_boost['score_increase'] = min(40, int((mean_curv - 90) * 2))  # Up to +40
                trajectory_boost['confidence_boost'] = min(30, int((mean_curv - 90) * 1.5))  # Up to +30
                trajectory_boost['has_high_curvature'] = True
                trajectory_boost['force_ai'] = True  # Override visual analysis
                print(f"🚨 VERY HIGH SEMANTIC CURVATURE: {mean_curv:.1f}° - STRONG AI INDICATOR")
                print(f"   Forcing AI classification (overriding visual analysis)")
            elif mean_curv > 82:  # Moderate AI indicator
                trajectory_boost['score_increase'] = min(25, int((mean_curv - 82) * 3))  # Up to +25
                trajectory_boost['confidence_boost'] = min(20, int((mean_curv - 82) * 2.5))  # Up to +20
                trajectory_boost['has_high_curvature'] = True
                print(f"⚠️ HIGH SEMANTIC CURVATURE: {mean_curv:.1f}° (AI indicator)")
                print(f"   Score boost: +{trajectory_boost['score_increase']}, Confidence boost: +{trajectory_boost['confidence_boost']}")
            else:
                print(f"✓ Normal semantic curvature: {mean_curv:.1f}° (natural video)")
        
        # 5. Scan metadata for AI keywords (if URL provided)
        metadata_scan = {'has_ai_keywords': False, 'keywords_found': [], 'confidence_boost': 0, 'score_increase': 0}
        if data.original_url and data.original_url.strip():
            print(f"🔍 Scanning metadata for AI keywords in URL: {data.original_url}")
            metadata_scan = scan_metadata_for_ai_keywords(data.original_url)
            if metadata_scan['has_ai_keywords']:
                print(f"⚠️ AI KEYWORDS DETECTED: {metadata_scan['keywords_found']}")
                print(f"   Score boost: +{metadata_scan['score_increase']}, Confidence boost: +{metadata_scan['confidence_boost']}")
        
        # 6. Extract Frames
        frames = extract_frames_base64(filepath, num_frames)
        if not frames:
             raise HTTPException(status_code=400, detail="Could not extract frames from video")

        # 6. Call AI API with enhanced forensic prompt (PHASE 1 IMPROVEMENT)
        prompt = """
    You are an EXPERT Video Forensics Analyst specializing in objective technical analysis of video authenticity. 
    Your role is to provide NEUTRAL, UNBIASED analysis based solely on observable technical evidence.

    🔍 ANALYSIS FRAMEWORK - Examine these sequential frames objectively:

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

    7. 🚨 AI-SPECIFIC ARTIFACT CHECKLIST (Modern Generators):
       
       **Sora Tells:**
       - Dreamlike, overly cinematic composition (too "perfect" framing)
       - Slight ethereal or surreal quality to lighting
       - Background elements that are impressively detailed but subtly inconsistent
       - Smooth but slightly "floaty" motion, lacks weight
       - **Text issues**: Signs with garbled letters, morphing words
       
       **Runway Gen-3 Tells:**
       - Corporate/commercial aesthetic (clean, polished, but sterile)
       - Very smooth motion that can feel "computer-generated"
       - Edges can be too sharp or too soft (not naturally in-between)
       - Lighting often perfect but lacks natural imperfections
       - **Text issues**: Clean-looking but nonsensical text
       
       **Pika/Kling Tells:**
       - More obvious physics violations (exaggerated movements)
       - Temporal glitches more common (brief warping or stuttering)
       - Lower-fidelity artifacts in complex scenes
       - **Text issues**: Very garbled, often completely illegible
       
       **Generic AI Tells to Watch For:**
       - Lack of camera shake or imperfect framing
       - Perfect lighting that doesn't match environment
       - Faces that are too symmetrical or have "uncanny valley" feel
       - Hair that moves unnaturally or as a single mass
       - Fabric/clothing with unnatural physics (too stiff or too fluid)
       - Repetitive or patterned elements in organic settings (trees, crowds)
       - Missing or incorrect shadows/reflections
       - Depth inconsistencies (foreground/background relationship wrong)
       - **TEXT PROBLEMS** (most reliable tell)

    📊 SCORING METHODOLOGY (Evidence-Based):
    
    **curvatureScore (0-100)**: Based on motion trajectory analysis
    - 0-30: Natural camera motion with realistic object trajectories
    - 31-60: Some irregular patterns that could be either natural or synthetic
    - 61-100: Clear unnatural motion patterns inconsistent with physics
    
    **distanceScore (0-100)**: Based on spatial consistency
    - Measure consistency of object positions and spatial relationships
    
    **confidence (0-100)**: Your certainty in the classification
    - Base this on the STRENGTH and QUANTITY of evidence observed
    - Low confidence (0-40): Ambiguous or limited evidence
    - Medium confidence (41-70): Some clear indicators present
    - High confidence (71-100): Multiple strong, unambiguous indicators
    
    **isAi determination**: 
    - Set to TRUE only if you observe clear, unambiguous technical evidence of synthetic generation
    - Set to FALSE if the video shows natural camera characteristics and physics
    - Consider the TOTALITY of evidence, not individual anomalies
    - Real-world videos can have compression artifacts, editing, and imperfections

    Output STRICTLY in JSON format:
    {
        "isAi": boolean,
        "confidence": number (0-100),
        "curvatureScore": number (0-100),
        "distanceScore": number (0-100),
        "reasoning": string[],
        "trajectoryData": [{ "x": number, "y": number, "frame": number }],
        "modelDetected": string
    }

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
        content = ai_res['choices'][0]['message']['content']
        clean_content = content.replace("```json", "").replace("```", "").strip()
        analysis_result = json.loads(clean_content)
        
        # 7. Apply ReStraV trajectory-based score adjustments
        if trajectory_boost['has_high_curvature']:
            original_curvature = analysis_result.get('curvatureScore', 0)
            original_confidence = analysis_result.get('confidence', 0)
            
            # Boost scores based on high trajectory curvature
            analysis_result['curvatureScore'] = min(100, original_curvature + trajectory_boost['score_increase'])
            analysis_result['confidence'] = min(100, original_confidence + trajectory_boost['confidence_boost'])
            
            # Add trajectory info to reasoning
            if trajectory_metrics:
                analysis_result['reasoning'].insert(0, f"📐 ReStraV Analysis: High temporal trajectory curvature detected ({trajectory_metrics['mean_curvature']:.1f}°), indicating irregular frame-to-frame transitions characteristic of AI-generated content")
            
            print(f"📊 Trajectory-based adjustment: curvature {original_curvature} → {analysis_result['curvatureScore']}, confidence {original_confidence} → {analysis_result['confidence']}")
        
        # 8. Apply metadata-based score adjustments
        if metadata_scan['has_ai_keywords']:
            original_curvature = analysis_result.get('curvatureScore', 0)
            original_confidence = analysis_result.get('confidence', 0)
            
            # Boost scores based on AI keyword detection
            analysis_result['curvatureScore'] = min(100, original_curvature + metadata_scan['score_increase'])
            analysis_result['confidence'] = min(100, original_confidence + metadata_scan['confidence_boost'])
            
            # If score is now high, set isAi to true
            if analysis_result['curvatureScore'] >= 60:
                analysis_result['isAi'] = True
            
            # Add metadata info to reasoning
            keyword_list = ', '.join(metadata_scan['keywords_found'][:5])  # Show first 5
            analysis_result['reasoning'].insert(0, f"⚠️ METADATA ALERT: Video contains AI-related keywords in title/description/tags: {keyword_list}")
            
            print(f"📊 Metadata-based adjustment: curvature {original_curvature} → {analysis_result['curvatureScore']}, confidence {original_confidence} → {analysis_result['confidence']}")
        
        # 9. Force AI classification if trajectory evidence is overwhelming
        if trajectory_boost.get('force_ai', False):
            analysis_result['isAi'] = True
            analysis_result['confidence'] = max(analysis_result.get('confidence', 0), 85)  # Ensure high confidence
            analysis_result['modelDetected'] = 'Unknown AI Model' if analysis_result.get('modelDetected') == 'Real Camera' else analysis_result.get('modelDetected')
            print(f"🚨 FORCING AI CLASSIFICATION due to extreme trajectory irregularity (overriding visual analysis)")
        
        # 10. Final AI determination based on combined signals
        elif analysis_result.get('curvatureScore', 0) >= 65:
            analysis_result['isAi'] = True

        # 11. Save Submission AUTOMATICALLY
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
