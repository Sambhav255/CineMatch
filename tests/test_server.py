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
# Tests for build_tmdb_params (Gemini output -> validated TMDB params)
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
# Tests for parse_tmdb_movies (TMDB Discover response -> movie list)
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
# Tests for parse_batch_explanations (Gemini output -> enrich movies)
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
