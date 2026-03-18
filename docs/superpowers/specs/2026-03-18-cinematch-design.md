# CineMatch — Full Functionality Design

**Date:** 2026-03-18
**Status:** Approved

## Goal

Make CineMatch fully functional and deployable to Vercel at zero cost. The app takes a natural language movie description (typed or spoken) and returns 3 matching movie recommendations with posters, ratings, synopses, and a "why this matches" explanation.

---

## Architecture

```
Browser (index.html)
  │
  ├── Voice input → Web Speech API (browser-native, free)
  ├── Text input → typed query
  │
  └── POST /api/recommend {"query": "..."}
        │
        └── server.py (FastAPI)
              │
              ├── 1. Gemini call → parse query into TMDB filter JSON
              ├── 2. TMDB Discover API → fetch matching movies
              ├── 3. Gemini call → generate "why this matches" for top 3
              └── 4. Return enriched movie cards to browser
```

### What is removed
- ChromaDB (vector database — too heavy for Vercel serverless, unnecessary with TMDB)
- sentence-transformers (local ML model — same reason)
- ElevenLabs transcription (replaced by browser Web Speech API)
- `build_index.py` (no longer needed)
- `transcribe_audio.py` (no longer needed)
- `/api/transcribe` endpoint (no longer needed)
- Debug logging code and `debug-444b48.log` writes
- Hardcoded ElevenLabs API key from `.env.example`

### What is added
- TMDB Discover API integration in `server.py`
- Gemini prompt for structured query parsing (outputs TMDB filter JSON)
- Web Speech API voice input in `index.html`
- `vercel.json` for Vercel deployment routing
- Startup validation with helpful setup instructions instead of crash

---

## API Keys Required

Both are free, no credit card needed.

| Service | URL | Free Limit |
|---------|-----|------------|
| Gemini API | https://aistudio.google.com | 1,500 req/day, 15 req/min |
| TMDB API | https://www.themoviedb.org/settings/api | Unlimited |

---

## Data Flow

### Step 1 — Gemini: Query Parser
**Input:** natural language user query
**Prompt:** instructs Gemini to output a JSON object with TMDB-compatible filters
**Output example:**
```json
{
  "genres": ["mystery", "comedy"],
  "min_rating": 6.5,
  "keywords": ["whodunit", "ensemble"],
  "sort_by": "vote_average.desc",
  "original_language": "en"
}
```

### Step 2 — TMDB Discover API
Uses the filter JSON to call `https://api.themoviedb.org/3/discover/movie`.
Returns up to 20 candidate movies with title, overview, poster path, rating, release date.
Top 3 are selected by vote average.

### Step 3 — Gemini: Explanation Writer
For each of the top 3 movies, calls Gemini with the user query + movie title + overview.
Returns a 1-2 sentence "why this matches" explanation.

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

### Voice Input (Web Speech API)
Replace the ElevenLabs audio recording flow with browser-native speech recognition:
- User clicks mic button → browser requests permission (one-time prompt)
- Web Speech API transcribes live, updating the text input in real time
- When user stops speaking, the transcript is auto-submitted
- Supported: Chrome, Edge, Safari. Firefox: mic button is hidden, text input shown.

### Visual changes
- No design changes — existing dark theme, gold accents, movie cards, skeleton loaders all stay
- Remove HTTPS/iframe warning banner (not needed for local or Vercel HTTPS)

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Missing API key on startup | Print clear setup instructions to console, return 503 with helpful message |
| Gemini rate limit (429) | Return user-facing: "Too many requests, try again in a moment" |
| TMDB returns 0 results | Return user-facing: "No movies found — try a different description" |
| Gemini returns malformed JSON | Retry once, then fall back to returning movies without match_reason |
| Voice not supported (Firefox) | Hide mic button, show text input only |
| No internet / API unreachable | Return user-facing: "Connection error, please check your internet" |

---

## Security

- Remove real ElevenLabs API key from `.env.example`, replace with `your_key_here` placeholder
- API keys read server-side only, never sent to browser
- `.env` remains in `.gitignore`
- CORS locked to localhost in development, configurable via `ALLOWED_ORIGINS` env var for Vercel

---

## Vercel Deployment

Add `vercel.json` at project root:
```json
{
  "builds": [{ "src": "server.py", "use": "@vercel/python" }],
  "routes": [{ "src": "/(.*)", "dest": "server.py" }]
}
```

Environment variables (Gemini key, TMDB key) set in Vercel dashboard — same names as `.env`.

---

## Files Changed

| File | Change |
|------|--------|
| `server.py` | Remove ChromaDB, ElevenLabs, debug logging; add TMDB Discover integration, Gemini query parser, better error handling |
| `index.html` | Replace ElevenLabs mic flow with Web Speech API |
| `.env.example` | Remove real API key, add TMDB_API_KEY, add setup comments |
| `requirements.txt` | Remove chromadb, sentence-transformers, elevenlabs, pandas |
| `vercel.json` | New file — Vercel routing config |
| `build_index.py` | Delete |
| `transcribe_audio.py` | Delete |

---

## Success Criteria

- User types or speaks a query and receives 3 movie recommendations with posters
- Each recommendation includes title, year, rating, synopsis, and match reason
- Voice input works in Chrome/Edge/Safari
- App runs locally with `python server.py`
- App deploys to Vercel without modification
- Zero paid services required
