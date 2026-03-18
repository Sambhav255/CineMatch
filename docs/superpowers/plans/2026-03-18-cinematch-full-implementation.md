# CineMatch Full Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken Gemini-hallucination + ElevenLabs architecture with a working TMDB-first + Web Speech API stack that runs locally and deploys to Vercel at zero cost.

**Architecture:** User query → Gemini parses it into TMDB Discover filter JSON (call 1) → TMDB Discover returns real movies → Gemini writes batch "why this matches" explanations for top 3 (call 2) → browser renders movie cards. Voice input handled entirely in-browser via Web Speech API.

**Tech Stack:** Python 3.12, FastAPI, `google-generativeai` (gemini-1.5-flash), `httpx`, TMDB Discover API, Web Speech API (browser), Vercel Python serverless

**Spec:** `docs/superpowers/specs/2026-03-18-cinematch-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `server.py` | Rewrite | FastAPI app: TMDB genre table, Gemini query parser, TMDB Discover call, Gemini batch explainer, lazy key validation |
| `public/index.html` | Create (moved from root) | SPA frontend with Web Speech API voice input |
| `requirements.txt` | Update | Remove chromadb, sentence-transformers, elevenlabs, pandas, python-multipart |
| `.env.example` | Update | Add TMDB_API_KEY, remove real ElevenLabs key |
| `vercel.json` | Create | Route /api/* to server.py, catch-all to public/ |
| `tests/test_server.py` | Create | Unit tests for parser helpers and endpoint behavior |
| `build_index.py` | Delete | No longer needed |
| `transcribe_audio.py` | Delete | No longer needed |

---

## Task 1: Clean Up Dead Files and Dependencies

**Files:**
- Delete: `build_index.py`
- Delete: `transcribe_audio.py`
- Modify: `requirements.txt`
- Modify: `.env.example`

- [ ] **Step 1: Delete dead files**

```bash
cd "/Users/sambhav/Library/CloudStorage/OneDrive-UniversityofSt.Thomas(2)/Desktop/Projects/CineMatch"
rm build_index.py transcribe_audio.py debug-444b48.log
```

- [ ] **Step 2: Rewrite requirements.txt**

Replace the entire file content with:

```
# CineMatch — AI Movie Finder
fastapi>=0.115.0,<1.0.0
uvicorn[standard]>=0.32.0,<1.0.0
google-generativeai>=0.8.0
httpx>=0.28.0
python-dotenv>=1.0.0
```

- [ ] **Step 3: Rewrite .env.example**

Replace the entire file content with:

```
# CineMatch environment variables
# Copy this file to .env and fill in your API keys.

# --- REQUIRED ---

# Google Gemini API key (free tier: 1,500 req/day, 15 req/min)
# Get yours at: https://aistudio.google.com
GEMINI_API_KEY=your_gemini_api_key_here

# TMDB (The Movie Database) API key (free, unlimited)
# Get yours at: https://www.themoviedb.org/settings/api
TMDB_API_KEY=your_tmdb_api_key_here

# --- OPTIONAL ---

# Gemini model to use (default: gemini-1.5-flash — fastest free-tier model)
GEMINI_MODEL_NAME=gemini-1.5-flash

# CORS origins (comma-separated). Defaults to http://localhost:3000 if unset.
# For Vercel: API and frontend share the same origin so this isn't needed.
# ALLOWED_ORIGINS=https://your-app.vercel.app
```

- [ ] **Step 4: Verify .venv still activates**

```bash
cd "/Users/sambhav/Library/CloudStorage/OneDrive-UniversityofSt.Thomas(2)/Desktop/Projects/CineMatch"
source .venv/bin/activate && pip install -r requirements.txt
```

Expected: packages install without error. `chromadb`, `sentence-transformers`, `elevenlabs` will be uninstalled or just not listed — that's fine.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: remove dead files, slim down requirements"
```

---

## Task 2: Write Tests for Server Logic

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_server.py`

These tests use mocks so they run without real API keys.

- [ ] **Step 1: Create tests directory**

```bash
mkdir -p "/Users/sambhav/Library/CloudStorage/OneDrive-UniversityofSt.Thomas(2)/Desktop/Projects/CineMatch/tests"
touch "/Users/sambhav/Library/CloudStorage/OneDrive-UniversityofSt.Thomas(2)/Desktop/Projects/CineMatch/tests/__init__.py"
```

- [ ] **Step 2: Write test file**

Create `tests/test_server.py` with this content:

```python
"""
Tests for CineMatch server logic.
Run with: pytest tests/ -v
All tests use mocks — no real API keys needed.
"""
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Provide dummy keys so server.py module loads without RuntimeError
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("TMDB_API_KEY", "test-tmdb-key")

# ---------------------------------------------------------------------------
# Tests for _clean_json_block
# ---------------------------------------------------------------------------

def test_clean_json_block_plain():
    from server import _clean_json_block
    raw = '[{"title": "A"}]'
    assert _clean_json_block(raw) == '[{"title": "A"}]'


def test_clean_json_block_strips_markdown_fences():
    from server import _clean_json_block
    raw = '```json\n[{"title": "A"}]\n```'
    result = _clean_json_block(raw)
    assert result.strip().startswith("[")
    assert '"title"' in result


def test_clean_json_block_strips_preamble():
    from server import _clean_json_block
    raw = 'Sure! Here is the JSON:\n[{"title": "A"}]\nHope that helps!'
    result = _clean_json_block(raw)
    assert result == '[{"title": "A"}]'


# ---------------------------------------------------------------------------
# Tests for build_tmdb_params (Gemini output → validated TMDB params)
# ---------------------------------------------------------------------------

def test_build_tmdb_params_valid():
    from server import build_tmdb_params
    gemini_output = '{"with_genres": "9648,35", "vote_average.gte": 6.5, "sort_by": "vote_average.desc"}'
    params = build_tmdb_params(gemini_output)
    assert params["with_genres"] == "9648,35"
    assert params["vote_average.gte"] == 6.5
    assert params["sort_by"] == "vote_average.desc"


def test_build_tmdb_params_with_markdown_fences():
    from server import build_tmdb_params
    gemini_output = '```json\n{"with_genres": "28"}\n```'
    params = build_tmdb_params(gemini_output)
    assert params["with_genres"] == "28"


def test_build_tmdb_params_malformed_returns_defaults():
    from server import build_tmdb_params
    params = build_tmdb_params("not valid json at all {{{{")
    # Should return safe defaults, not raise
    assert isinstance(params, dict)
    assert "sort_by" in params


# ---------------------------------------------------------------------------
# Tests for parse_tmdb_movies (TMDB Discover response → movie list)
# ---------------------------------------------------------------------------

def test_parse_tmdb_movies_extracts_top_3():
    from server import parse_tmdb_movies
    tmdb_response = {
        "results": [
            {"id": 1, "title": "Movie A", "release_date": "2019-06-01",
             "vote_average": 8.1, "overview": "Synopsis A",
             "poster_path": "/a.jpg", "genre_ids": [9648, 35]},
            {"id": 2, "title": "Movie B", "release_date": "2018-01-15",
             "vote_average": 7.5, "overview": "Synopsis B",
             "poster_path": "/b.jpg", "genre_ids": [28]},
            {"id": 3, "title": "Movie C", "release_date": "2020-11-20",
             "vote_average": 9.0, "overview": "Synopsis C",
             "poster_path": "/c.jpg", "genre_ids": [18]},
            {"id": 4, "title": "Movie D", "release_date": "2021-03-05",
             "vote_average": 6.0, "overview": "Synopsis D",
             "poster_path": "/d.jpg", "genre_ids": [53]},
        ]
    }
    movies = parse_tmdb_movies(tmdb_response)
    assert len(movies) == 3
    # Top 3 by vote_average: C (9.0), A (8.1), B (7.5)
    assert movies[0]["title"] == "Movie C"
    assert movies[1]["title"] == "Movie A"
    assert movies[2]["title"] == "Movie B"


def test_parse_tmdb_movies_builds_poster_url():
    from server import parse_tmdb_movies
    tmdb_response = {
        "results": [
            {"id": 1, "title": "Movie A", "release_date": "2019-06-01",
             "vote_average": 8.0, "overview": "Synopsis",
             "poster_path": "/abc.jpg", "genre_ids": [28]},
        ]
    }
    movies = parse_tmdb_movies(tmdb_response)
    assert movies[0]["imageUrl"] == "https://image.tmdb.org/t/p/w500/abc.jpg"


def test_parse_tmdb_movies_handles_missing_poster():
    from server import parse_tmdb_movies
    tmdb_response = {
        "results": [
            {"id": 1, "title": "Movie A", "release_date": "2019-06-01",
             "vote_average": 8.0, "overview": "Synopsis",
             "poster_path": None, "genre_ids": []},
        ]
    }
    movies = parse_tmdb_movies(tmdb_response)
    assert movies[0]["imageUrl"] is None


def test_parse_tmdb_movies_returns_empty_on_no_results():
    from server import parse_tmdb_movies
    movies = parse_tmdb_movies({"results": []})
    assert movies == []


def test_parse_tmdb_movies_formats_genre_names():
    from server import parse_tmdb_movies
    tmdb_response = {
        "results": [
            {"id": 1, "title": "Movie A", "release_date": "2019-06-01",
             "vote_average": 7.0, "overview": "Synopsis",
             "poster_path": None, "genre_ids": [28, 35]},
        ]
    }
    movies = parse_tmdb_movies(tmdb_response)
    assert "Action" in movies[0]["genre"]
    assert "Comedy" in movies[0]["genre"]


def test_parse_tmdb_movies_formats_rating():
    from server import parse_tmdb_movies
    tmdb_response = {
        "results": [
            {"id": 1, "title": "Movie A", "release_date": "2019-06-01",
             "vote_average": 7.9, "overview": "Synopsis",
             "poster_path": None, "genre_ids": []},
        ]
    }
    movies = parse_tmdb_movies(tmdb_response)
    assert movies[0]["rating"] == "7.9/10"


# ---------------------------------------------------------------------------
# Tests for parse_batch_explanations (Gemini output → enrich movies)
# ---------------------------------------------------------------------------

def test_parse_batch_explanations_enriches_movies():
    from server import parse_batch_explanations
    movies = [
        {"title": "A", "year": 2019, "rating": "8.0/10", "synopsis": "S", "genre": "Drama", "imageUrl": None},
        {"title": "B", "year": 2018, "rating": "7.5/10", "synopsis": "S", "genre": "Action", "imageUrl": None},
        {"title": "C", "year": 2020, "rating": "7.0/10", "synopsis": "S", "genre": "Comedy", "imageUrl": None},
    ]
    gemini_output = json.dumps([
        {"why": "Why A", "mood": "Dark", "director": "Dir A", "duration": "2h", "streamingNote": "Netflix"},
        {"why": "Why B", "mood": "Action", "director": "Dir B", "duration": "1h 45min", "streamingNote": "Check streaming"},
        {"why": "Why C", "mood": "Fun", "director": "Dir C", "duration": "1h 30min", "streamingNote": "HBO"},
    ])
    result = parse_batch_explanations(gemini_output, movies)
    assert result[0]["why"] == "Why A"
    assert result[0]["director"] == "Dir A"
    assert result[1]["mood"] == "Action"
    assert result[2]["streamingNote"] == "HBO"


def test_parse_batch_explanations_graceful_on_malformed():
    from server import parse_batch_explanations
    movies = [
        {"title": "A", "year": 2019, "rating": "8.0/10", "synopsis": "S", "genre": "Drama", "imageUrl": None},
    ]
    result = parse_batch_explanations("not json {{{", movies)
    # Should return movies with fallback values, not raise
    assert len(result) == 1
    assert result[0]["title"] == "A"
    assert "why" in result[0]


# ---------------------------------------------------------------------------
# Tests for /api/recommend endpoint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_recommend_returns_3_movies():
    """End-to-end test with all external calls mocked."""
    from httpx import AsyncClient, ASGITransport
    from server import app

    fake_tmdb_filter = '{"with_genres": "18", "vote_average.gte": 7.0, "sort_by": "vote_average.desc"}'
    fake_tmdb_response = {
        "results": [
            {"id": i, "title": f"Movie {i}", "release_date": "2019-01-01",
             "vote_average": 8.0 - i * 0.1, "overview": f"Synopsis {i}",
             "poster_path": f"/{i}.jpg", "genre_ids": [18]}
            for i in range(5)
        ]
    }
    fake_explanations = json.dumps([
        {"why": f"Why {i}", "mood": "Dramatic", "director": f"Director {i}",
         "duration": "2h", "streamingNote": "Netflix"}
        for i in range(3)
    ])

    mock_gemini_parse = MagicMock()
    mock_gemini_parse.text = fake_tmdb_filter

    mock_gemini_explain = MagicMock()
    mock_gemini_explain.text = fake_explanations

    mock_model = MagicMock()
    mock_model.generate_content.side_effect = [mock_gemini_parse, mock_gemini_explain]

    mock_tmdb_resp = MagicMock()
    mock_tmdb_resp.raise_for_status = MagicMock()
    mock_tmdb_resp.json.return_value = fake_tmdb_response

    with patch("server.genai.GenerativeModel", return_value=mock_model), \
         patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_tmdb_resp)
        mock_client_cls.return_value = mock_client

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post("/api/recommend", json={"query": "a dramatic film"})

    assert response.status_code == 200
    data = response.json()
    assert len(data["movies"]) == 3
    assert data["movies"][0]["title"] == "Movie 0"


@pytest.mark.asyncio
async def test_recommend_returns_empty_with_message_when_no_tmdb_results():
    from httpx import AsyncClient, ASGITransport
    from server import app

    mock_gemini_parse = MagicMock()
    mock_gemini_parse.text = '{"with_genres": "28"}'

    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_gemini_parse

    mock_tmdb_resp = MagicMock()
    mock_tmdb_resp.raise_for_status = MagicMock()
    mock_tmdb_resp.json.return_value = {"results": []}

    with patch("server.genai.GenerativeModel", return_value=mock_model), \
         patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_tmdb_resp)
        mock_client_cls.return_value = mock_client

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post("/api/recommend", json={"query": "xyzzy gibberish"})

    assert response.status_code == 200
    data = response.json()
    assert data["movies"] == []
    assert "No movies found" in data.get("message", "")


@pytest.mark.asyncio
async def test_recommend_returns_400_on_empty_query():
    from httpx import AsyncClient, ASGITransport
    from server import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/api/recommend", json={"query": "   "})

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_recommend_returns_503_when_gemini_key_missing():
    from httpx import AsyncClient, ASGITransport
    import server
    original = server.GEMINI_API_KEY
    server.GEMINI_API_KEY = None
    try:
        from server import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post("/api/recommend", json={"query": "a film"})
        assert response.status_code == 503
    finally:
        server.GEMINI_API_KEY = original


@pytest.mark.asyncio
async def test_recommend_returns_503_when_tmdb_key_missing():
    from httpx import AsyncClient, ASGITransport
    import server
    original = server.TMDB_API_KEY
    server.TMDB_API_KEY = None
    try:
        from server import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post("/api/recommend", json={"query": "a film"})
        assert response.status_code == 503
    finally:
        server.TMDB_API_KEY = original
```

- [ ] **Step 3: Install pytest-asyncio and create pytest.ini**

```bash
cd "/Users/sambhav/Library/CloudStorage/OneDrive-UniversityofSt.Thomas(2)/Desktop/Projects/CineMatch"
source .venv/bin/activate && pip install pytest pytest-asyncio
```

Create `pytest.ini` at the project root:

```ini
[pytest]
asyncio_mode = auto
```

This is required for `pytest-asyncio` ≥0.21 — without it, async tests fail in strict mode.

- [ ] **Step 4: Run tests — verify they all FAIL (server.py not yet rewritten)**

```bash
cd "/Users/sambhav/Library/CloudStorage/OneDrive-UniversityofSt.Thomas(2)/Desktop/Projects/CineMatch"
source .venv/bin/activate && python -m pytest tests/ -v 2>&1 | head -60
```

Expected: ImportError or failures on `build_tmdb_params`, `parse_tmdb_movies`, `parse_batch_explanations` — these functions don't exist yet. That's correct for TDD.

- [ ] **Step 5: Commit tests**

```bash
git add tests/
git commit -m "test: add failing tests for new TMDB-first architecture"
```

---

## Task 3: Rewrite server.py

**Files:**
- Modify: `server.py` (full rewrite)

- [ ] **Step 1: Replace server.py with the new implementation**

Write the following content to `server.py`:

```python
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
    """Strip markdown fences and extract the first JSON array or object."""
    text = text.strip()
    # Try to find a JSON array first, then object
    for pattern in (r"\[.*\]", r"\{.*\}"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(0)
    # Fallback: strip markdown fences
    for fence in ("```json", "```JSON", "```"):
        text = text.replace(fence, "")
    return text.strip()


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
5. "streamingNote": best guess at where it streams (e.g. "Available on Netflix", "Available on HBO Max") or "Check streaming availability"

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
    tmdb_params.setdefault("vote_count.gte", 100)

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
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=503, detail=f"Connection error, please check your internet.")
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
        # Graceful degradation: return movies without explanations
        explain_response = None

    if explain_response:
        enriched = parse_batch_explanations(explain_response.text or "", movies)
    else:
        enriched = parse_batch_explanations("", movies)  # triggers fallback

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
```

- [ ] **Step 2: Run tests — verify they now PASS**

```bash
cd "/Users/sambhav/Library/CloudStorage/OneDrive-UniversityofSt.Thomas(2)/Desktop/Projects/CineMatch"
source .venv/bin/activate && python -m pytest tests/ -v
```

Expected: All tests pass. If any fail, read the error message and fix the relevant function in `server.py` before continuing.

- [ ] **Step 3: Commit**

```bash
git add server.py
git commit -m "feat: rewrite server with TMDB-first architecture and lazy key validation"
```

---

## Task 4: Update Frontend — Web Speech API + Move to public/

**Files:**
- Create: `public/index.html` (moved from root, with changes)
- No changes to CSS or card rendering — only the voice input section changes

- [ ] **Step 1: Create public/ directory**

```bash
mkdir -p "/Users/sambhav/Library/CloudStorage/OneDrive-UniversityofSt.Thomas(2)/Desktop/Projects/CineMatch/public"
```

- [ ] **Step 2: Create public/index.html**

Start from `index.html` and apply these changes:

1. **Remove** the HTTPS/iframe banner `<div id="httpsBanner" ...>` block (lines 323–327)
2. **Remove** the banner detection JavaScript IIFE block (the `(function() { const inIframe...})();` block, lines 451–464)
3. **Remove** `style="padding-top:0"` from `<div class="app" ...>` (line 329)
4. **Replace** the entire `// ── Voice via MediaRecorder → server transcription ──` section (from `let mediaRecorder = null;` through the closing of `toggleVoice`, `stopListening`, and `setListeningUI` functions, lines 478–579) with the Web Speech API implementation below
5. **Update** the "Voice Ready" status text in the header to reflect browser support
6. **Update** `renderResults` to handle empty movies array with a "no results" message
7. **Update** `doSearch` to show the `message` field from the response when movies is empty

**Replacement for the voice section (replaces lines 478–579):**

```javascript
// ── Voice via Web Speech API (browser-native, no server call) ────────────────
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const speechSupported = !!SpeechRecognition;
let recognition = null;

// Hide mic button on unsupported browsers (Firefox)
if (!speechSupported) {
  $('micBtn').style.display = 'none';
  $('statusText').textContent = 'Type to search';
}

$('micBtn').addEventListener('click', toggleVoice);

function toggleVoice() {
  if (isListening) {
    stopListening();
    return;
  }
  if (!speechSupported) return;
  clearError();

  recognition = new SpeechRecognition();
  recognition.lang = 'en-US';
  recognition.interimResults = true;
  recognition.maxAlternatives = 1;

  recognition.onstart = () => {
    isListening = true;
    setListeningUI(true);
    $('transcriptText').textContent = 'Listening…';
    $('voiceIndicator').classList.add('show');
  };

  recognition.onresult = (event) => {
    let interim = '';
    let final = '';
    for (let i = event.resultIndex; i < event.results.length; i++) {
      if (event.results[i].isFinal) {
        final += event.results[i][0].transcript;
      } else {
        interim += event.results[i][0].transcript;
      }
    }
    const current = final || interim;
    if (current) {
      ta.value = current;
      ta.style.height = 'auto';
      ta.style.height = ta.scrollHeight + 'px';
      $('findBtn').disabled = false;
      $('transcriptText').textContent = current;
    }
  };

  recognition.onend = () => {
    $('voiceIndicator').classList.remove('show');
    setListeningUI(false);
    isListening = false;
    const text = ta.value.trim();
    if (text) {
      doSearch(text);
    }
  };

  recognition.onerror = (event) => {
    $('voiceIndicator').classList.remove('show');
    setListeningUI(false);
    isListening = false;
    if (event.error === 'not-allowed') {
      showError('Microphone permission denied. Please allow mic access in your browser settings.');
    } else if (event.error === 'no-speech') {
      showError('No speech detected. Try speaking louder or closer to your mic.');
    } else if (event.error !== 'aborted') {
      showError('Voice error: ' + event.error + '. Try typing instead.');
    }
  };

  recognition.start();
}

function stopListening() {
  if (recognition) {
    recognition.stop();
    recognition = null;
  }
  isListening = false;
  setListeningUI(false);
}

function setListeningUI(on) {
  $('micBtn').classList.toggle('active', on);
  $('micIcon').style.display = on ? 'none' : '';
  $('waveIcon').style.display = on ? 'block' : 'none';
  $('inputWrap').classList.toggle('listening', on);
  $('inputHint').textContent = on ? 'Tap mic to stop' : 'Press Enter or click Find →';
}
```

**Update to `doSearch` — handle empty movies + message field:**

Replace the `renderResults(data.movies, q);` line in `doSearch` with:

```javascript
if (data.movies && data.movies.length > 0) {
  renderResults(data.movies, q);
} else {
  showError(data.message || 'No movies found — try a different description.');
}
```

**Update header status span** — add `id="statusText"` to the `<span>Voice Ready</span>` element so JavaScript can update it:

```html
<span id="statusText">Voice Ready</span>
```

- [ ] **Step 3: Verify the file was created correctly**

```bash
ls -la "/Users/sambhav/Library/CloudStorage/OneDrive-UniversityofSt.Thomas(2)/Desktop/Projects/CineMatch/public/"
```

Expected: `index.html` present.

- [ ] **Step 4: Remove the old root-level index.html**

```bash
rm "/Users/sambhav/Library/CloudStorage/OneDrive-UniversityofSt.Thomas(2)/Desktop/Projects/CineMatch/index.html"
```

- [ ] **Step 5: Commit**

```bash
git add public/
git rm index.html
git commit -m "feat: move frontend to public/, replace ElevenLabs with Web Speech API"
```

---

## Task 5: Add vercel.json

**Files:**
- Create: `vercel.json`

- [ ] **Step 1: Create vercel.json**

```json
{
  "builds": [
    { "src": "server.py", "use": "@vercel/python" }
  ],
  "routes": [
    { "src": "/api/(.*)", "dest": "server.py" },
    { "src": "/(.*)", "dest": "/public/$1" }
  ]
}
```

- [ ] **Step 2: Commit**

```bash
git add vercel.json
git commit -m "chore: add vercel.json for deployment"
```

---

## Task 6: Local Smoke Test

- [ ] **Step 1: Set up .env with real keys**

```bash
cp "/Users/sambhav/Library/CloudStorage/OneDrive-UniversityofSt.Thomas(2)/Desktop/Projects/CineMatch/.env.example" \
   "/Users/sambhav/Library/CloudStorage/OneDrive-UniversityofSt.Thomas(2)/Desktop/Projects/CineMatch/.env"
```

Then open `.env` and fill in real values for `GEMINI_API_KEY` and `TMDB_API_KEY`.

Get keys from:
- Gemini: https://aistudio.google.com (click "Get API key")
- TMDB: https://www.themoviedb.org/settings/api (register free account → API → create key)

- [ ] **Step 2: Start the server**

```bash
cd "/Users/sambhav/Library/CloudStorage/OneDrive-UniversityofSt.Thomas(2)/Desktop/Projects/CineMatch"
source .venv/bin/activate && python server.py
```

Expected: `Uvicorn running on http://0.0.0.0:3000`

- [ ] **Step 3: Test the API directly**

In a new terminal:

```bash
curl -s -X POST http://localhost:3000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "a feel-good 90s comedy"}' | python -m json.tool | head -40
```

Expected: JSON response with `movies` array of 3 items, each with title, year, rating, synopsis, why, mood, director, duration, streamingNote, imageUrl.

- [ ] **Step 4: Test in browser**

Open http://localhost:3000 — should see the CineMatch UI.
Type a query and click Find. Should get 3 movie cards with posters.
In Chrome/Edge: click the mic button, speak a query. Should transcribe and search.

- [ ] **Step 5: Test error cases**

```bash
# Empty query → 400
curl -s -X POST http://localhost:3000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": ""}' | python -m json.tool

# No-results query → 200 with empty movies + message
curl -s -X POST http://localhost:3000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "xyzzy florp bloop moop"}' | python -m json.tool
```

- [ ] **Step 6: Run full test suite one final time**

```bash
python -m pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 7: Final commit**

```bash
git add .env.example  # make sure the cleaned .env.example is committed, NOT .env
git status  # verify .env is NOT listed (it should be gitignored)
git commit -m "chore: verify local smoke test passes"
```

---

## Task 7: Vercel Deployment (when ready)

- [ ] **Step 1: Push to GitHub**

Create a repo at github.com, then:

```bash
git remote add origin https://github.com/YOUR_USERNAME/cinematch.git
git push -u origin main
```

- [ ] **Step 2: Connect to Vercel**

1. Go to vercel.com → New Project → Import from GitHub → select your repo
2. Framework preset: **Other**
3. Root directory: leave as `/`
4. Add environment variables:
   - `GEMINI_API_KEY` = your key
   - `TMDB_API_KEY` = your key
5. Deploy

- [ ] **Step 3: Verify deployment**

Open the Vercel URL:
- `https://your-app.vercel.app/` — should serve the UI
- Test a query → should return movie cards with posters

---

## Done Criteria

- [ ] All unit tests pass (`pytest tests/ -v`)
- [ ] `python server.py` starts without error (even without .env)
- [ ] A typed query returns 3 movie cards with posters locally
- [ ] Voice input works in Chrome/Edge
- [ ] Firefox shows text input only (no broken mic button)
- [ ] Empty query returns 400 with a readable message
- [ ] No API keys in git history
