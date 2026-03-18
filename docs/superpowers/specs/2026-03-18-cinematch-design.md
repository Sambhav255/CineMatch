# CineMatch — Full Functionality Design

**Date:** 2026-03-18
**Status:** Approved

## Goal

Make CineMatch fully functional and deployable to Vercel at zero cost. The app takes a natural language movie description (typed or spoken) and returns 3 matching movie recommendations with posters, ratings, synopses, and a "why this matches" explanation.

---

## Architecture

```
Browser (index.html — served as static file from public/)
  │
  ├── Voice input → Web Speech API (browser-native, free)
  ├── Text input → typed query
  │
  └── POST /api/recommend {"query": "..."}
        │
        └── server.py (FastAPI)
              │
              ├── 1. Gemini call → parse query into TMDB filter JSON (genre IDs, not names)
              ├── 2. TMDB Discover API → fetch matching movies
              └── 3. Gemini call → generate "why this matches" for all 3 in one batch call
              └── 4. Return enriched movie cards to browser
```

### What is replaced (not extended)
The existing `call_gemini_recommend` function — which asks Gemini to hallucinate movie titles — is **deleted entirely** and replaced with the TMDB-first flow described above. Nothing from the old recommendation path is preserved.

### What is removed
- ChromaDB (vector database — too heavy for Vercel serverless, unnecessary with TMDB)
- sentence-transformers (local ML model — same reason)
- ElevenLabs transcription (replaced by browser Web Speech API)
- `build_index.py` (no longer needed)
- `transcribe_audio.py` (no longer needed)
- `/api/transcribe` endpoint (no longer needed)
- Debug logging code and `debug-444b48.log` writes
- Hardcoded ElevenLabs API key from `.env.example`
- `python-multipart` from `requirements.txt` (only needed for file upload endpoint, which is removed)

### What is added
- TMDB Discover API integration in `server.py`
- Hardcoded TMDB genre ID lookup table (e.g. `{"comedy": 35, "thriller": 53, ...}`) — avoids extra API call
- Gemini prompt for structured query parsing (outputs TMDB filter JSON with numeric genre IDs)
- Gemini prompt for batched explanation generation (all 3 movies in one call)
- Web Speech API voice input in `index.html`
- `public/index.html` — move index.html to `public/` for reliable Vercel static serving
- `vercel.json` for Vercel deployment routing
- Lazy API key validation (in request handler, not at module load) with helpful error messages

---

## API Keys Required

Both are free, no credit card needed.

| Service | URL | Free Limit |
|---------|-----|------------|
| Gemini API (`gemini-1.5-flash`) | https://aistudio.google.com | 1,500 req/day, 15 req/min |
| TMDB API | https://www.themoviedb.org/settings/api | Unlimited |

---

## Data Flow

### Step 1 — Gemini: Query Parser (1 API call)
**Input:** natural language user query
**Prompt:** instructs Gemini to output a JSON object with TMDB-compatible filters using numeric genre IDs
**Genre resolution:** a hardcoded lookup table in `server.py` maps common genre names to TMDB IDs (no extra API call needed). The Gemini prompt includes this table so it outputs IDs directly.
**No keywords field** — TMDB keyword filtering requires numeric IDs resolved via a separate API call per keyword; this complexity is not worth the marginal benefit. Dropped from design.

**Output example:**
```json
{
  "with_genres": "9648,35",
  "vote_average.gte": 6.5,
  "sort_by": "vote_average.desc",
  "with_original_language": "en"
}
```

### Step 2 — TMDB Discover API (1 API call)
Uses the filter JSON directly as query params to `https://api.themoviedb.org/3/discover/movie`.
Returns up to 20 candidate movies with title, overview, poster path, rating, release date.
**Top 3 are always selected by `vote_average` descending**, regardless of the `sort_by` value Gemini emits. Gemini's `sort_by` is passed to TMDB for initial filtering only; final selection is always by quality (vote average) to prevent low-rated popular movies from surfacing.

### Step 3 — Gemini: Batched Explanation Writer (1 API call total)
All 3 movie explanations are generated in a **single Gemini call** to stay well within the 15 req/min free tier limit.

**Total Gemini calls per user request: 2** (query parse + batch explanation)

**Prompt includes:** user query + title + overview for all 3 movies
**Output:** JSON array of 3 match_reason strings

### Step 4 — Response to Browser
```json
{
  "query": "original user query",
  "movies": [
    {
      "title": "Knives Out",
      "year": "2019",
      "rating": 7.9,
      "poster": "https://image.tmdb.org/t/p/w500/...",
      "synopsis": "...",
      "match_reason": "Matches your cozy mystery vibe with its witty ensemble cast..."
    }
  ]
}
```

---

## Frontend Changes

### Static File Serving
`index.html` moves to `public/index.html`. On Vercel, the `public/` directory is served as static assets — this is more reliable than `FileResponse` from a Python serverless function where the working directory is not guaranteed. Locally, FastAPI serves it via `StaticFiles` mount.

### Voice Input (Web Speech API)
Replace the ElevenLabs audio recording flow with browser-native speech recognition:
- User clicks mic button → browser requests permission (one-time prompt)
- Web Speech API transcribes live, updating the text input in real time
- When user stops speaking, the transcript is auto-submitted
- Supported: Chrome, Edge, Safari. Firefox: mic button is hidden, text input shown with a note.

### Visual changes
- No design changes — existing dark theme, gold accents, movie cards, skeleton loaders all stay
- Remove HTTPS/iframe warning banner (not needed for local or Vercel HTTPS)

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Missing GEMINI_API_KEY | Lazy check in request handler — return 503 with message: "Server not configured: missing GEMINI_API_KEY. See .env.example for setup instructions." |
| Missing TMDB_API_KEY | Same lazy check — return 503 with message: "Server not configured: missing TMDB_API_KEY." |
| Empty/blank query | Return 400: "Please enter a movie description." |
| Gemini rate limit (429) | Return 429 to client: "Too many requests, try again in a moment." |
| TMDB returns 0 results | Return 200 with empty array + message: "No movies found — try a different description." |
| Gemini returns malformed JSON | Attempt to strip markdown fences and re-parse (no extra API call). If still unparseable, return movies without match_reason (graceful degradation). No second Gemini call — total remains 2 per request. |
| Voice not supported (Firefox) | Hide mic button, show text input only with no error |
| Network/API unreachable | Return 503: "Connection error, please check your internet." |

Key validation is **lazy** (inside the request handler or FastAPI startup event), not at module load time. This prevents the process from crashing before it can return a useful HTTP response.

---

## Security

- Remove real ElevenLabs API key from `.env.example`, replace with `your_key_here` placeholder
- API keys read server-side only, never sent to browser
- `.env` remains in `.gitignore`
- CORS: On Vercel, the static frontend and API share the same origin — CORS headers are irrelevant for same-origin requests. Locally, default to `["http://localhost:3000"]` when `ALLOWED_ORIGINS` is unset. The `ALLOWED_ORIGINS` env var is available for overriding (e.g., if testing from a different port or `file://`). Do NOT default to `*` — use the explicit localhost default to avoid surprises.

---

## Vercel Deployment

`index.html` is in `public/` — Vercel serves it as a static file automatically.

`vercel.json` at project root:
```json
{
  "builds": [{ "src": "server.py", "use": "@vercel/python" }],
  "routes": [
    { "src": "/api/(.*)", "dest": "server.py" },
    { "src": "/(.*)", "dest": "/public/$1" }
  ]
}
```

The catch-all route `/(.*) → /public/$1` is required. When both `builds` and `routes` are present, Vercel's default static routing is overridden — without this explicit route, `/` returns 404 instead of serving `public/index.html`.

Environment variables (`GEMINI_API_KEY`, `TMDB_API_KEY`) set in Vercel dashboard — same names as `.env`.

---

## Files Changed

| File | Change |
|------|--------|
| `server.py` | Remove ChromaDB, ElevenLabs, debug logging, old Gemini hallucination flow; add TMDB Discover integration, genre ID lookup table, batched Gemini explanation call, lazy key validation, better error handling |
| `index.html` → `public/index.html` | Move to public/, replace ElevenLabs mic flow with Web Speech API, remove HTTPS banner |
| `.env.example` | Remove real API key, add `TMDB_API_KEY`, add setup comments pointing to free signup URLs |
| `requirements.txt` | Remove: chromadb, sentence-transformers, elevenlabs, pandas, python-multipart |
| `vercel.json` | New file — Vercel routing config |
| `build_index.py` | Delete |
| `transcribe_audio.py` | Delete |

---

## Success Criteria

**Functional:**
- User types a query and receives exactly 3 movie recommendations
- Each recommendation includes title, year, rating, poster image, synopsis, and match reason
- Voice input (Chrome/Edge/Safari) transcribes speech into the text box and auto-submits
- Firefox shows text input only (no broken mic button)

**Error cases:**
- Empty query returns a 400 with a user-readable message (not a 500)
- Starting server without API keys prints a clear setup message and returns 503 on requests (does not crash on startup)
- A query with no TMDB results (e.g., "xyzzy gibberish") returns a 200 with an empty array and the message "No movies found — try a different description"

**Local:**
- `python server.py` starts successfully
- App is accessible at `http://localhost:3000`

**Vercel:**
- `index.html` loads at the Vercel deployment URL
- A test query returns 3 movie cards with posters
- Environment variables set in Vercel dashboard are read correctly by the API
