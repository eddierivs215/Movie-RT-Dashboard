from __future__ import annotations

import math
from datetime import datetime
from typing import Optional, Tuple

# Ranking weights (with RT available)
W_RT = 0.65
W_AUDIENCE = 0.30
W_EVIDENCE = 0.05
# Ranking weights (without RT)
W_AUDIENCE_NO_RT = 0.80
W_EVIDENCE_NO_RT = 0.20
# Confidence thresholds: vote counts at which confidence reaches 1.0
TMDB_CONFIDENCE_CAP = 5000
IMDB_CONFIDENCE_CAP = 50000


def runtime_bucket_to_bounds(bucket: str) -> Tuple[Optional[int], Optional[int]]:
    b = (bucket or "").strip()
    if b == "< 1 hour":
        return (0, 59)
    if b == "1–2 hours":
        return (60, 119)
    if b == "> 2 hours":
        return (120, None)
    return (None, None)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _safe_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        # NaN check
        if f != f:
            return None
        return f
    except Exception:
        return None


def _safe_int(v) -> Optional[int]:
    try:
        if v is None:
            return None
        i = int(v)
        return i
    except Exception:
        # handle strings like "12,345"
        try:
            s = str(v).replace(",", "").strip()
            return int(s) if s.isdigit() else None
        except Exception:
            return None


def rank_score(
    rt: Optional[int],
    tmdb_vote: Optional[float],
    tmdb_votes: Optional[int],
    imdb_rating: Optional[float] = None,
    imdb_votes: Optional[int] = None,
) -> float:
    """
    Rank score in [0, 1] (roughly), higher is better.

    Design goals:
    - Keep RT (critic) as primary when present.
    - Use audience signals (TMDB + IMDb) as secondary / stabilizing.
    - Use vote counts as confidence weights, but avoid letting huge vote counts dominate.

    Inputs:
    - rt: Rotten Tomatoes % (0–100) or None
    - tmdb_vote: TMDB average (0–10) or None
    - tmdb_votes: TMDB vote count or None
    - imdb_rating: IMDb rating (0–10) or None
    - imdb_votes: IMDb vote count or None
    """

    rt_f = _safe_float(rt)
    tmdb_f = _safe_float(tmdb_vote)
    tmdb_n = _safe_int(tmdb_votes)
    imdb_f = _safe_float(imdb_rating)
    imdb_n = _safe_int(imdb_votes)

    # Normalize to 0..1
    rt_norm = _clamp((rt_f / 100.0)) if rt_f is not None else None
    tmdb_norm = _clamp((tmdb_f / 10.0)) if tmdb_f is not None else None
    imdb_norm = _clamp((imdb_f / 10.0)) if imdb_f is not None else None

    # Confidence weights from vote counts (0..1), capped to prevent domination
    tmdb_conf = _clamp((tmdb_n or 0) / TMDB_CONFIDENCE_CAP)
    imdb_conf = _clamp((imdb_n or 0) / IMDB_CONFIDENCE_CAP)

    # Audience composite (0..1). If one source missing, fall back to the other.
    if tmdb_norm is None and imdb_norm is None:
        audience = 0.0
    elif tmdb_norm is None:
        audience = imdb_norm
    elif imdb_norm is None:
        audience = tmdb_norm
    else:
        # Weighted by confidence, with a small floor so both can matter early.
        w_tmdb = 0.20 + 0.80 * tmdb_conf
        w_imdb = 0.20 + 0.80 * imdb_conf
        audience = (w_tmdb * tmdb_norm + w_imdb * imdb_norm) / (w_tmdb + w_imdb)

    # Overall confidence bonus: prefer titles with more audience evidence
    evidence = 0.5 * tmdb_conf + 0.5 * imdb_conf

    # Primary scoring logic
    if rt_norm is not None:
        # RT-first, audience as secondary
        base = W_RT * rt_norm + W_AUDIENCE * audience + W_EVIDENCE * evidence
    else:
        # No RT available: rely on audience more heavily
        base = W_AUDIENCE_NO_RT * audience + W_EVIDENCE_NO_RT * evidence

    return _clamp(base)


# ------------------------------------------------------------------
# Repetition penalty constants
# ------------------------------------------------------------------
RECENCY_WINDOW_DAYS = 14
FREQUENCY_WEIGHT = 0.3

# Discovery score constants
OBSCURITY_WEIGHT = 0.5
FRESHNESS_WEIGHT = 0.4
FRESHNESS_WINDOW_DAYS = 30
DISCOVERY_QUALITY_FLOOR = 0.55
DISCOVERY_POPULARITY_CAP = 2000
DISCOVERY_MIN_VOTES = 30
POPULARITY_LOG_SCALE = 10000


def repetition_penalty(shown_count: int, last_shown_at: Optional[str]) -> float:
    """
    Returns multiplier in [0, 1]. 1.0 = no penalty.
    recency  = min(1.0, days_elapsed / RECENCY_WINDOW_DAYS)
    frequency = 1.0 / (1.0 + log(1 + shown_count) * FREQUENCY_WEIGHT)
    """
    if shown_count == 0 or last_shown_at is None:
        return 1.0
    try:
        days = (datetime.now() - datetime.fromisoformat(last_shown_at)).days
    except (ValueError, TypeError):
        days = RECENCY_WINDOW_DAYS
    recency = min(1.0, days / RECENCY_WINDOW_DAYS)
    frequency = 1.0 / (1.0 + math.log(1 + shown_count) * FREQUENCY_WEIGHT)
    return recency * frequency


def discovery_score(
    base_rank_score: float,
    tmdb_votes: Optional[int],
    shown_count: int = 0,
    last_shown_at: Optional[str] = None,
) -> float:
    """
    Secondary score for Hidden Gems lane.
    Returns 0.0 if below quality floor or outside popularity band.
    Upper bound ~2.0 (used only for relative ranking within gem pool).
    """
    votes = tmdb_votes or 0
    if base_rank_score < DISCOVERY_QUALITY_FLOOR:
        return 0.0
    if votes > DISCOVERY_POPULARITY_CAP or votes < DISCOVERY_MIN_VOTES:
        return 0.0

    # Log-scaled obscurity: penalizes popularity on a log curve
    pop_norm = math.log(votes + 1) / math.log(POPULARITY_LOG_SCALE + 1)
    obscurity = 1.0 + OBSCURITY_WEIGHT * (1.0 - pop_norm)

    # Freshness bonus: higher if not recently shown
    if shown_count == 0 or last_shown_at is None:
        freshness = 1.0 + FRESHNESS_WEIGHT
    else:
        try:
            days = (datetime.now() - datetime.fromisoformat(last_shown_at)).days
        except (ValueError, TypeError):
            days = FRESHNESS_WINDOW_DAYS
        freshness = 1.0 + FRESHNESS_WEIGHT * min(1.0, days / FRESHNESS_WINDOW_DAYS)

    return base_rank_score * obscurity * freshness
