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
            # Resize to save token usage/bandwidth
            # Maintain aspect ratio, max width 512
            height, width = frame.shape[:2]
            MAX_WIDTH = 512
            if width > MAX_WIDTH:
                scale = MAX_WIDTH / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            b64 = base64.b64encode(buffer).decode('utf-8')
            frames.append(f"data:image/jpeg;base64,{b64}")
            extracted += 1
            
        count += 1

    cap.release()
    return frames

# Endpoints
@app.post("/upload")
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

@app.post("/download")
async def download_video(request: DownloadRequest):
    try:
        print(f"📥 Download request received for URL: {request.url}")
        
        file_id = str(uuid.uuid4())
        filename = f"{file_id}.%(ext)s"  # Let yt-dlp determine the extension
        filepath = os.path.join(DOWNLOAD_DIR, f"{file_id}")
        
        ydl_opts = {
            'format': 'best[height<=720]/best',
            'outtmpl': filepath + '.%(ext)s',
            'quiet': False,  # Enable output for debugging
            'no_warnings': False,
            'max_filesize': 100 * 1024 * 1024,  # Increase to 100MB
            'noplaylist': True,
            'geo_bypass': True,
            # Add headers and options to avoid 403 errors
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-us,en;q=0.5',
                'Accept-Encoding': 'gzip,deflate',
                'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.7',
                'Connection': 'keep-alive',
            },
            'extractor_args': {
                'youtube': {
                    'skip': ['dash', 'hls'],
                    'player_skip': ['configs'],
                }
            },
            'cookiefile': None,
            'age_limit': None,
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
        # 3. Extract Frames
        frames = extract_frames_base64(filepath, 15)
        if not frames:
             raise HTTPException(status_code=400, detail="Could not extract frames from video")

        # 4. Call AI API
        prompt = """
    You are a specialized AI Video Forensics Engine based on the "FrameTruth" (Representation Straightening) methodology.
    Your task is to detect AI-generated videos by identifying "temporal curvature" anomalies and specific generative artifacts.

    ANALYSIS MODE: You are provided with 15 sequential frames extracted from the video.

    Step 1: Scan for specific AI Artifacts:
    - Texture Boiling/Flickering
    - Object Permanence Failures
    - Physics Violations (Gravity, Lighting, Momentum)
    - Background Incoherence
    - Anomalous Morphing

    Step 2: Evaluate based on FrameTruth Method:
    - Natural Video: Smooth, straight latent trajectory.
    - AI Video: Erratic, jagged trajectory.

    Output strictly in JSON format matching this schema:
    {
        "isAi": boolean,
        "confidence": number (0-100),
        "curvatureScore": number (0-100),
        "distanceScore": number (0-100),
        "reasoning": string[],
        "trajectoryData": [{ "x": number, "y": number, "frame": number }],
        "modelDetected": string
    }

    IMPORTANT INSTRUCTIONS:
    1. "reasoning": Provide CLEAR, SIMPLE explanations suitable for a general audience. Avoid jargon.
    2. "trajectoryData": You MUST populate this with 15-20 coordinate points simulating the motion path.
    3. "modelDetected": Best guess of the source. If it looks like a real video, set this to "Real Camera". If it looks AI but specific model is unknown, set to "Unknown AI Model". DO NOT return "N/A".
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
    app.mount("/", StaticFiles(directory="dist", html=True), name="frontend")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
