from typing import Optional, List
import logging
import time
import requests

logger = logging.getLogger(__name__)

TMDB_BASE = "https://api.themoviedb.org/3"
MAX_RETRIES = 3
RETRY_BACKOFF = 1.0  # seconds


def _tmdb_request(url: str, params: dict) -> Optional[dict]:
    """Make a TMDB API request with retry logic and error handling."""
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=30)
        except requests.RequestException:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF * (attempt + 1))
                continue
            logger.warning("TMDB request failed after %d retries: %s", MAX_RETRIES, url)
            return None

        if r.status_code != 200:
            if attempt < MAX_RETRIES - 1 and r.status_code >= 500:
                time.sleep(RETRY_BACKOFF * (attempt + 1))
                continue
            logger.warning("TMDB returned status %d for %s", r.status_code, url)
            return None

        try:
            return r.json()
        except ValueError:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF * (attempt + 1))
                continue
            logger.warning("TMDB returned invalid JSON for %s", url)
            return None

    return None


def tmdb_get_genres(api_key: str) -> dict[str, int]:
    data = _tmdb_request(
        f"{TMDB_BASE}/genre/movie/list",
        params={"api_key": api_key, "language": "en-US"},
    )
    if data is None:
        return {}
    return {g["name"]: g["id"] for g in data.get("genres", [])}


def tmdb_discover_movies(
    api_key: str,
    genre_ids: List[int],
    runtime_min: Optional[int],
    runtime_max: Optional[int],
    year_min: Optional[int],
    year_max: Optional[int],
    page: int = 1,
) -> dict:
    params = {
        "api_key": api_key,
        "language": "en-US",
        "sort_by": "popularity.desc",
        "include_adult": "false",
        "include_video": "false",
        "page": page,
        "with_genres": ",".join(map(str, genre_ids)) if genre_ids else None,
        "with_runtime.gte": runtime_min,
        "with_runtime.lte": runtime_max,
        "primary_release_date.gte": f"{year_min}-01-01" if year_min else None,
        "primary_release_date.lte": f"{year_max}-12-31" if year_max else None,
    }
    params = {k: v for k, v in params.items() if v is not None}

    data = _tmdb_request(f"{TMDB_BASE}/discover/movie", params=params)
    if data is None:
        return {"results": []}
    return data


def tmdb_movie_details(api_key: str, movie_id: int) -> dict:
    data = _tmdb_request(
        f"{TMDB_BASE}/movie/{movie_id}",
        params={"api_key": api_key, "language": "en-US"},
    )
    if data is None:
        return {}
    return data
