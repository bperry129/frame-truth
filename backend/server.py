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

# Verify API key is loaded
if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY not found in .env file!")
    print("Please check that backend/.env exists and contains: OPENROUTER_API_KEY=your_key_here")
else:
    print(f"✓ API Key loaded: {OPENROUTER_API_KEY[:20]}...")

app = FastAPI()

ADMIN_USER = "admin"
ADMIN_PASS = "admin123"

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
    # Whitelist localhost
    if ip == "127.0.0.1" or ip == "localhost":
        return True

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Rate limit: 5 per day
    one_day_ago = (datetime.now() - timedelta(days=1)).isoformat()
    c.execute("SELECT count(*) FROM submissions WHERE ip_address = ? AND created_at > ?", (ip, one_day_ago))
    count = c.fetchone()[0]
    conn.close()
    return count < 5

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
            "url": f"http://localhost:8000/videos/{filename}"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Upload failed: {str(e)}"})

@app.post("/download")
async def download_video(request: DownloadRequest):
    try:
        file_id = str(uuid.uuid4())
        filename = f"{file_id}.mp4"
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        
        ydl_opts = {
            'format': 'best', # Let yt-dlp decide best quality (cv2 matches most formats)
            'outtmpl': filepath,
            'quiet': True,
            'no_warnings': True,
            'max_filesize': 50 * 1024 * 1024, # 50MB
            'noplaylist': True,
            'geo_bypass': True,
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        
        meta_info = {}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info and download
            try:
                info = ydl.extract_info(request.url, download=True)
                meta_info = {
                    "title": info.get("title"),
                    "uploader": info.get("uploader"),
                    "duration": info.get("duration"),
                    "view_count": info.get("view_count")
                }
            except Exception as e:
                # Fallback if extraction fails but download might work?
                # Usually extract_info throws if download fails.
                raise e

        if not os.path.exists(filepath):
             return JSONResponse(status_code=400, content={"detail": "Download failed: File not created"})
        
        return {
            "filename": filename,
            "url": f"http://localhost:8000/videos/{filename}",
            "meta": meta_info
        }

        if not os.path.exists(filepath):
             return JSONResponse(status_code=400, content={"detail": "Download failed: File not created"})
        
        return {
            "filename": filename,
            "url": f"http://localhost:8000/videos/{filename}",
            "meta": meta_info
        }

    except Exception as e:
        return JSONResponse(status_code=400, content={"detail": f"Download failed: {str(e)}"})

@app.post("/api/analyze")
async def analyze_video(request: Request, data: AnalyzeRequest):
    # 1. Check Rate Limit
    client_ip = request.headers.get("x-forwarded-for", request.client.host).split(",")[0]
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

@app.get("/docs")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
