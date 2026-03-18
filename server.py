import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import google.generativeai as genai
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent

app = FastAPI()

# ── CORS ─────────────────────────────────────────────────────────────────────
_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()
ALLOWED_ORIGINS = (
    [o.strip() for o in _origins_env.split(",") if o.strip()]
    if _origins_env
    else ["http://localhost:3000"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files (local dev) ─────────────────────────────────────────────────
_public_dir = PROJECT_ROOT / "public"
if _public_dir.is_dir():
    app.mount("/public", StaticFiles(directory=str(_public_dir)), name="public")

# ── Config (read lazily in handlers, not at module load) ─────────────────────
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
TMDB_API_KEY: Optional[str] = os.getenv("TMDB_API_KEY")
GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# Configure Gemini if key is present (lazy — won't crash on missing key)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ── TMDB genre lookup ─────────────────────────────────────────────────────────
TMDB_GENRE_IDS: Dict[str, int] = {
    "action": 28, "adventure": 12, "animation": 16, "comedy": 35,
    "crime": 80, "documentary": 99, "drama": 18, "family": 10751,
    "fantasy": 14, "history": 36, "horror": 27, "music": 10402,
    "mystery": 9648, "romance": 10749, "science fiction": 878, "sci-fi": 878,
    "thriller": 53, "war": 10752, "western": 37,
}
TMDB_GENRE_NAMES: Dict[int, str] = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
    80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
    14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
    9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
    53: "Thriller", 10752: "War", 37: "Western",
}

# ── JSON helpers ──────────────────────────────────────────────────────────────

def _clean_json_block(text: str) -> str:
    """Strip markdown fences and extract the first complete JSON array or object."""
    text = text.strip()
    # Strip markdown fences first
    for fence in ("```json", "```JSON", "```"):
        text = text.replace(fence, "")
    text = text.strip()
    # Find first complete JSON array (greedy is fine here — arrays rarely nest weirdly)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return match.group(0)
    # Find first complete JSON object using balanced brace counting (avoids greedy over-match)
    start = text.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return text


def build_tmdb_params(gemini_json_str: str) -> Dict[str, Any]:
    """Parse Gemini's filter JSON into TMDB Discover query params.
    Returns safe defaults if parsing fails."""
    defaults: Dict[str, Any] = {
        "sort_by": "vote_average.desc",
        "vote_count.gte": 100,
        "with_original_language": "en",
    }
    try:
        cleaned = _clean_json_block(gemini_json_str)
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            return defaults
        # Only allow known safe TMDB Discover params
        allowed_keys = {
            "with_genres", "vote_average.gte", "vote_average.lte",
            "sort_by", "primary_release_date.gte", "primary_release_date.lte",
            "with_original_language",
        }
        params = {k: v for k, v in data.items() if k in allowed_keys}
        return {**defaults, **params}
    except (json.JSONDecodeError, ValueError):
        return defaults


def parse_tmdb_movies(tmdb_response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert TMDB Discover response into a list of movie dicts, top 3 by vote_average."""
    results = tmdb_response.get("results") or []
    if not results:
        return []
    # Sort by vote_average descending, take top 3
    sorted_results = sorted(results, key=lambda m: m.get("vote_average", 0), reverse=True)
    top3 = sorted_results[:3]
    movies = []
    for m in top3:
        genre_ids: List[int] = m.get("genre_ids") or []
        genre_str = " / ".join(TMDB_GENRE_NAMES[gid] for gid in genre_ids if gid in TMDB_GENRE_NAMES) or "Film"
        release_date: str = m.get("release_date") or ""
        year = int(release_date[:4]) if release_date and release_date[:4].isdigit() else 0
        vote_avg = m.get("vote_average") or 0.0
        poster_path = m.get("poster_path")
        movies.append({
            "title": m.get("title", "Unknown"),
            "year": year,
            "genre": genre_str,
            "rating": f"{vote_avg:.1f}/10",
            "synopsis": m.get("overview") or "",
            "imageUrl": f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else None,
            "_tmdb_id": m.get("id"),
        })
    return movies


def parse_batch_explanations(
    gemini_json_str: str, movies: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Merge Gemini's batch explanation JSON into the movie list.
    Falls back gracefully if parsing fails."""
    fallback_fields = {
        "why": "A great match for your taste.",
        "mood": "Engaging",
        "director": "Unknown",
        "duration": "Unknown",
        "streamingNote": "Check streaming availability",
    }
    try:
        cleaned = _clean_json_block(gemini_json_str)
        explanations = json.loads(cleaned)
        if not isinstance(explanations, list):
            raise ValueError("Expected a JSON array")
    except (json.JSONDecodeError, ValueError):
        return [{**m, **fallback_fields} for m in movies]

    enriched = []
    for i, movie in enumerate(movies):
        if i < len(explanations) and isinstance(explanations[i], dict):
            exp = explanations[i]
            enriched.append({
                **movie,
                "why": exp.get("why") or fallback_fields["why"],
                "mood": exp.get("mood") or fallback_fields["mood"],
                "director": exp.get("director") or fallback_fields["director"],
                "duration": exp.get("duration") or fallback_fields["duration"],
                "streamingNote": exp.get("streamingNote") or fallback_fields["streamingNote"],
            })
        else:
            enriched.append({**movie, **fallback_fields})
    return enriched


# ── Gemini prompts ────────────────────────────────────────────────────────────

def _build_filter_prompt(query: str) -> str:
    genre_table = ", ".join(f'"{name}": {gid}' for name, gid in TMDB_GENRE_IDS.items())
    return f"""You are a movie recommendation assistant. Convert the user's movie query into TMDB Discover API filters.

Available genre IDs: {{{genre_table}}}

Return ONLY a JSON object with these optional fields:
- "with_genres": comma-separated TMDB genre IDs as a string, e.g. "28,12"
- "vote_average.gte": minimum rating as a number between 0 and 10
- "primary_release_date.gte": earliest release date as "YYYY-01-01"
- "primary_release_date.lte": latest release date as "YYYY-12-31"
- "with_original_language": ISO 639-1 language code, e.g. "en"
- "sort_by": TMDB sort field, e.g. "vote_average.desc" or "popularity.desc"

User query: "{query}"

Return ONLY the JSON object. No explanation, no markdown."""


def _build_explanation_prompt(query: str, movies: List[Dict[str, Any]]) -> str:
    movies_json = json.dumps(
        [{"title": m["title"], "year": m["year"], "synopsis": m["synopsis"]} for m in movies],
        indent=2,
        ensure_ascii=False,
    )
    return f"""The user wants to watch: "{query}"

For each of the following movies, provide:
1. "why": 1-2 sentences explaining exactly why this movie matches the user's request
2. "mood": 2-4 word mood tag (e.g. "Dark & Thrilling", "Heartwarming & Fun", "Mind-Bending")
3. "director": the director's full name
4. "duration": runtime formatted as "Xh Ymin" (e.g. "2h 15min") or "Xh" if no minutes
5. "streamingNote": if you are confident the movie is on a major platform (Netflix, HBO Max, Disney+, Prime Video, Hulu), say "Available on [platform]". If unsure, say "Check streaming availability" — do NOT guess.

Movies:
{movies_json}

Return ONLY a JSON array of exactly {len(movies)} objects, each with keys: why, mood, director, duration, streamingNote.
No explanation, no markdown."""


# ── Request / Response models ─────────────────────────────────────────────────

class Movie(BaseModel):
    title: str
    year: int
    genre: str
    director: str
    rating: str
    duration: str
    synopsis: str
    why: str
    mood: str
    streamingNote: str
    imageUrl: Optional[str] = None


class RecommendRequest(BaseModel):
    query: str


class RecommendResponse(BaseModel):
    query: str
    movies: List[Movie]
    message: Optional[str] = None


# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.post("/api/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    # Lazy key validation
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Server not configured: missing GEMINI_API_KEY. See .env.example for setup instructions.",
        )
    if not TMDB_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Server not configured: missing TMDB_API_KEY. See .env.example for setup instructions.",
        )

    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Please enter a movie description.")

    model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    # Step 1: Parse query into TMDB filters
    try:
        filter_response = model.generate_content(_build_filter_prompt(query))
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "quota" in error_str.lower():
            raise HTTPException(status_code=429, detail="Too many requests, try again in a moment.")
        raise HTTPException(status_code=503, detail="Connection error, please check your internet.")

    tmdb_params = build_tmdb_params(filter_response.text or "")
    tmdb_params["api_key"] = TMDB_API_KEY

    # Step 2: Fetch movies from TMDB Discover
    try:
        async with httpx.AsyncClient() as client:
            tmdb_resp = await client.get(
                f"{TMDB_BASE_URL}/discover/movie",
                params=tmdb_params,
                timeout=10.0,
            )
            tmdb_resp.raise_for_status()
            tmdb_data = tmdb_resp.json()
    except httpx.HTTPStatusError:
        raise HTTPException(status_code=503, detail="Connection error, please check your internet.")
    except Exception:
        raise HTTPException(status_code=503, detail="Connection error, please check your internet.")

    movies = parse_tmdb_movies(tmdb_data)

    if not movies:
        return RecommendResponse(
            query=query,
            movies=[],
            message="No movies found — try a different description.",
        )

    # Step 3: Batch explanation (single Gemini call for all 3)
    try:
        explain_response = model.generate_content(_build_explanation_prompt(query, movies))
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "quota" in error_str.lower():
            raise HTTPException(status_code=429, detail="Too many requests, try again in a moment.")
        explain_response = None

    enriched = parse_batch_explanations(
        explain_response.text if explain_response else "", movies
    )

    # Remove internal field before building response
    for m in enriched:
        m.pop("_tmdb_id", None)

    return RecommendResponse(
        query=query,
        movies=[Movie(**m) for m in enriched],
    )


# ── Static frontend ───────────────────────────────────────────────────────────

@app.get("/")
async def index():
    index_path = PROJECT_ROOT / "public" / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="index.html not found in public/")
    return FileResponse(index_path, media_type="text/html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
