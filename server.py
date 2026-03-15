import json
import os

import google.generativeai as genai
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
try:
    from transcribe_audio import transcribe_audio
    TRANSCRIBE_AVAILABLE = True
except ImportError:
    transcribe_audio = None
    TRANSCRIBE_AVAILABLE = False

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── LLM configuration (Gemini) ──────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")


def _clean_json_block(text: str) -> str:
    cleaned = text.strip()
    for fence in ("```json", "```JSON", "```"):
        cleaned = cleaned.replace(fence, "")
    return cleaned.strip()


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
            detail="Voice transcription unavailable (missing pplx/llm_api). Use text input.",
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

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
