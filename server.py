import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

try:
    from transcribe_audio import transcribe_audio
    TRANSCRIBE_AVAILABLE = True
except ImportError:
    transcribe_audio = None
    TRANSCRIBE_AVAILABLE = False

# Resolve project root for serving static files and index.html
PROJECT_ROOT = Path(__file__).resolve().parent

app = FastAPI()

# Configurable CORS origins (e.g. ALLOWED_ORIGINS=https://app.example.com,https://www.example.com or * for dev)
_origins = os.getenv("ALLOWED_ORIGINS", "*").strip()
ALLOWED_ORIGINS = ["*"] if _origins == "*" else [o.strip() for o in _origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static assets if present (e.g. CSS, JS, images)
_static_dir = PROJECT_ROOT / "static"
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# ── LLM configuration (Gemini) ──────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")


def _clean_json_block(text: str) -> str:
    """Extract the JSON array from LLM output: first '[' to last ']', ignoring conversational filler."""
    text = text.strip()
    # Regex: match from first '[' to last ']' (DOTALL so newlines inside array are included)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return match.group(0)
    # Fallback: strip markdown code fences and return as-is
    for fence in ("```json", "```JSON", "```"):
        text = text.replace(fence, "")
    return text.strip()


def _parse_movies_from_text(text: str):
    cleaned = _clean_json_block(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not parse movie recommendations. Please try again. ({e})",
        )
    if not isinstance(data, list):
        raise HTTPException(
            status_code=500,
            detail="Model returned an unexpected format (expected a JSON array of movies).",
        )
    return data


def _build_recommendation_prompt(query: str) -> str:
    return f"""
You are CineMatch, an expert movie recommender with encyclopedic knowledge.
The user described what they want to watch: \"{query}\"

Recommend exactly 3 movies.
Return ONLY a valid JSON array (no markdown, no explanation), in this shape:
[
  {{
    "title": "Movie Title",
    "year": 1994,
    "genre": "Drama / Crime",
    "director": "Director Name",
    "rating": "8.9/10",
    "duration": "2h 22min",
    "synopsis": "A compelling 2-3 sentence description that captures the essence.",
    "why": "1-2 sentences explaining exactly why this matches what the user described.",
    "mood": "Dark & Thrilling",
    "streamingNote": "Available on Netflix / or Check streaming availability"
  }}
]

Rules:
- Always return exactly 3 movie objects.
- Do not include any text before or after the JSON.
"""


def call_gemini_recommend(query: str):
    prompt = _build_recommendation_prompt(query)
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    try:
        response = model.generate_content(prompt)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Problem contacting the recommendation model. Please try again. ({e})",
        )
    text = response.text or ""
    movies = _parse_movies_from_text(text)
    return movies

class RecommendRequest(BaseModel):
    query: str

@app.post("/api/recommend")
async def recommend(req: RecommendRequest):
    try:
        movies = call_gemini_recommend(req.query)
        return {"movies": movies, "query": req.query}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not TRANSCRIBE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Voice transcription unavailable (missing elevenlabs). Use text input.",
        )
    try:
        audio_bytes = await file.read()
        # Detect media type from content_type or default to webm
        media_type = file.content_type or "audio/webm"
        # Normalize — ElevenLabs accepts audio/webm, audio/mp4, audio/wav, audio/ogg
        if "webm" in media_type:
            media_type = "audio/webm"
        elif "mp4" in media_type or "m4a" in media_type:
            media_type = "audio/mp4"
        elif "ogg" in media_type:
            media_type = "audio/ogg"
        elif "wav" in media_type:
            media_type = "audio/wav"
        else:
            media_type = "audio/webm"
        result = await transcribe_audio(audio_bytes, media_type=media_type)
        return {"text": result["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def index():
    index_path = PROJECT_ROOT / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path, media_type="text/html")


# Production: run with `uvicorn server:app` or `gunicorn server:app -k uvicorn.workers.UvicornWorker`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
