from typing import Optional, Tuple
from pathlib import Path
import json
import logging
import time
import requests

logger = logging.getLogger(__name__)

OMDB_CACHE_FILE = Path(__file__).with_name(".omdb_cache.json")
MAX_RETRIES = 3
RETRY_BACKOFF = 1.0  # seconds
MAX_CACHE_ENTRIES = 5000

# In-memory cache loaded once at module import
_cache: dict = {}
_cache_dirty = False


def _load_cache() -> None:
    global _cache
    if OMDB_CACHE_FILE.exists():
        try:
            _cache = json.loads(OMDB_CACHE_FILE.read_text())
        except (json.JSONDecodeError, IOError):
            _cache = {}
    else:
        _cache = {}


def flush_cache() -> None:
    """Write accumulated cache changes to disk."""
    global _cache_dirty
    if not _cache_dirty:
        return
    try:
        OMDB_CACHE_FILE.write_text(json.dumps(_cache, indent=2))
        _cache_dirty = False
    except IOError:
        logger.warning("Failed to flush OMDb cache to disk")


def get_cache_size() -> int:
    return len(_cache)


# Load cache on module import
_load_cache()


def omdb_lookup_by_imdb_id(api_key: str, imdb_id: str, use_cache: bool = True, media_type: str = "movie") -> Optional[dict]:
    global _cache, _cache_dirty

    if not imdb_id:
        return None

    # Check in-memory cache first
    if use_cache and imdb_id in _cache:
        return _cache[imdb_id]

    # Retry loop for transient failures
    data = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(
                "https://www.omdbapi.com/",
                params={"apikey": api_key, "i": imdb_id, "type": media_type, "r": "json"},
                timeout=30,
            )
        except requests.RequestException:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF * (attempt + 1))
                continue
            return None

        # If key is invalid / rate-limited / unauthorized, don't crash the app
        if r.status_code != 200:
            if attempt < MAX_RETRIES - 1 and r.status_code >= 500:
                time.sleep(RETRY_BACKOFF * (attempt + 1))
                continue
            return None

        try:
            data = r.json()
            break  # success
        except ValueError:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF * (attempt + 1))
                continue
            return None

    if data is None:
        return None

    if data.get("Response") != "True":
        # Check for rate limit error
        if "limit" in data.get("Error", "").lower():
            return {"_rate_limited": True}
        return None

    # Add to in-memory cache (evict oldest if over limit)
    if use_cache:
        if len(_cache) >= MAX_CACHE_ENTRIES:
            oldest_key = next(iter(_cache))
            del _cache[oldest_key]
        _cache[imdb_id] = data
        _cache_dirty = True

    return data


def extract_rotten_tomatoes_score(omdb_payload: dict) -> Optional[int]:
    ratings = omdb_payload.get("Ratings") or []
    for item in ratings:
        if item.get("Source") == "Rotten Tomatoes":
            val = (item.get("Value") or "").strip()
            if val.endswith("%"):
                try:
                    return int(val[:-1])
                except ValueError:
                    return None
    return None


def extract_imdb_score(omdb_payload: dict) -> Tuple[Optional[float], Optional[int]]:
    """Extract IMDb rating and vote count from OMDb payload.

    Returns:
        (rating, votes) where rating is 0-10 float and votes is int count.
        Either or both may be None if not available.
    """
    if not omdb_payload:
        return None, None

    rating = None
    votes = None

    raw_rating = omdb_payload.get("imdbRating")
    if raw_rating and raw_rating != "N/A":
        try:
            rating = float(raw_rating)
        except ValueError:
            pass

    raw_votes = omdb_payload.get("imdbVotes")
    if raw_votes and raw_votes != "N/A":
        try:
            votes = int(str(raw_votes).replace(",", ""))
        except ValueError:
            pass

    return rating, votes
