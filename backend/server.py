import os
import uvicorn
import sqlite3
import uuid
import json
import cv2
import base64
import requests
import shutil
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import yt_dlp

load_dotenv(".env")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS")

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
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

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
            
            # Critical: Use Android client to bypass bot detection
            'extractor_args': {
                'youtube': {
                    # Force Android client (most reliable)
                    'player_client': ['android'],
                    'skip': ['dash', 'hls'],
                    'player_skip': ['webpage', 'configs'],
                }
            },
            
            # Enhanced headers mimicking mobile Chrome
            'http_headers': {
                'User-Agent': 'com.google.android.youtube/19.09.37 (Linux; U; Android 11) gzip',
                'Accept': '*/*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate',
                'X-YouTube-Client-Name': '3',
                'X-YouTube-Client-Version': '19.09.37',
            },
            
            # Additional anti-bot measures
            'sleep_interval': 1,
            'max_sleep_interval': 5,
            'retries': 10,
            'fragment_retries': 10,
            'file_access_retries': 5,
            
            # Network settings
            'socket_timeout': 30,
            'source_address': None,
            
            # Avoid potential issues
            'nocheckcertificate': True,
            'prefer_insecure': False,
        }
        
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
                    "view_count": info.get("view_count")
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
        error_msg = f"Download failed: {str(e)}"
        print(f"❌ Error: {error_msg}")
        return JSONResponse(status_code=400, content={"detail": error_msg})

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
        
        # PHASE 1 IMPROVEMENT: Adaptive frame count (50-100 frames for better accuracy)
        if duration < 10:
            num_frames = 50  # Very short clips (e.g., TikTok, Instagram Reels)
            print(f"📹 Very short video ({duration:.1f}s) - using 50 frames for maximum accuracy")
        elif duration < 30:
            num_frames = 75  # Short videos (Shorts, brief clips)
            print(f"📹 Short video ({duration:.1f}s) - using 75 frames for high accuracy")
        elif duration < 60:
            num_frames = 100  # Medium videos
            print(f"📹 Medium video ({duration:.1f}s) - using 100 frames for comprehensive analysis")
        else:
            num_frames = 100  # Longer videos - cap at 100 to balance cost/accuracy
            print(f"📹 Long video ({duration:.1f}s) - using 100 frames (capped)")
        
        # 4. Extract Frames
        frames = extract_frames_base64(filepath, num_frames)
        if not frames:
             raise HTTPException(status_code=400, detail="Could not extract frames from video")

        # 5. Call AI API with enhanced forensic prompt (PHASE 1 IMPROVEMENT)
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

    6. 🚨 AI-SPECIFIC ARTIFACT CHECKLIST (Modern Generators):
       
       **Sora Tells:**
       - Dreamlike, overly cinematic composition (too "perfect" framing)
       - Slight ethereal or surreal quality to lighting
       - Background elements that are impressively detailed but subtly inconsistent
       - Smooth but slightly "floaty" motion, lacks weight
       
       **Runway Gen-3 Tells:**
       - Corporate/commercial aesthetic (clean, polished, but sterile)
       - Very smooth motion that can feel "computer-generated"
       - Edges can be too sharp or too soft (not naturally in-between)
       - Lighting often perfect but lacks natural imperfections
       
       **Pika/Kling Tells:**
       - More obvious physics violations (exaggerated movements)
       - Temporal glitches more common (brief warping or stuttering)
       - Lower-fidelity artifacts in complex scenes
       - Text/small details often garbled or morphing
       
       **Generic AI Tells to Watch For:**
       - Lack of camera shake or imperfect framing
       - Perfect lighting that doesn't match environment
       - Faces that are too symmetrical or have "uncanny valley" feel
       - Hair that moves unnaturally or as a single mass
       - Fabric/clothing with unnatural physics (too stiff or too fluid)
       - Repetitive or patterned elements in organic settings (trees, crowds)
       - Missing or incorrect shadows/reflections
       - Depth inconsistencies (foreground/background relationship wrong)

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
                 "model": "google/gemini-2.0-flash-001",
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
