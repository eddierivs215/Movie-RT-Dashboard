from __future__ import annotations

import os
import logging
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from tmdb import tmdb_get_genres, tmdb_discover_movies, tmdb_movie_details
from omdb import omdb_lookup_by_imdb_id, extract_rotten_tomatoes_score, extract_imdb_score, get_cache_size, flush_cache
from recommend import runtime_bucket_to_bounds, rank_score, W_RT, W_AUDIENCE, W_EVIDENCE, W_AUDIENCE_NO_RT, W_EVIDENCE_NO_RT

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
SEEN_FILE = Path(__file__).with_name("seen_movies.csv")

# Page depth constants
SURPRISE_EXTRA_PAGES = 5  # Additional pages fetched when Surprise is enabled
MAX_TOTAL_PAGES = 10      # Hard cap on total pages
MAX_OVERVIEW_LENGTH = 240
NEAR_MISS_RT_OFFSET = 15
NEAR_MISS_RT_FLOOR = 50
NEAR_MISS_TMDB_FLOOR = 7.0
SURPRISE_TMDB_FLOOR = 6.5
RUNTIME_SHORT_MAX = 100   # minutes; below this = "short"
RUNTIME_MEDIUM_MAX = 140  # minutes; above this = "long"
CURRENT_YEAR = date.today().year

# Diversity & surprise constants
RATING_TIER_ACCLAIMED = 90
RATING_TIER_SOLID = 75
RATING_TIER_MIXED = 60
POPULARITY_BLOCKBUSTER = 5000
POPULARITY_POPULAR = 1000
DEFAULT_SURPRISE_PICKS = 5
DIVERSITY_WEIGHT_GENRE = 2
DIVERSITY_WEIGHT_DEFAULT = 1
WILDCARD_TMDB_FLOOR = 6.0


def load_seen() -> set[str]:
    if SEEN_FILE.exists():
        df_seen = pd.read_csv(SEEN_FILE, encoding="utf-8")
        if "Title" in df_seen.columns:
            return set(df_seen["Title"].dropna().astype(str).tolist())
    return set()


def save_seen(seen_set: set[str]) -> bool:
    try:
        pd.DataFrame({"Title": sorted(seen_set)}).to_csv(SEEN_FILE, index=False)
        return True
    except (IOError, OSError) as e:
        logger.warning("Failed to save seen movies: %s", e)
        return False


# ------------------------------------------------------------------
# Diversity sampling helpers
# ------------------------------------------------------------------
SKIP_TITLE_WORDS = {"the", "a", "an", "el", "la", "le", "les", "der", "die", "das", "il", "lo"}


def get_title_key(title: str) -> str:
    if not title:
        return ""
    words = title.lower().split()
    for word in words:
        clean = "".join(c for c in word if c.isalnum())
        if clean and clean not in SKIP_TITLE_WORDS and len(clean) > 1:
            return clean[:5]
    return title[:5].lower() if title else ""


def get_decade(release_year: str | None) -> str:
    if not release_year:
        return "unk"
    try:
        year_str = str(release_year)[:4]
        year = int(year_str)
        return f"{(year // 10) * 10}s"
    except (ValueError, TypeError):
        return "unk"


def get_runtime_category(runtime: int | float | None) -> str:
    if runtime is None or pd.isna(runtime):
        return "unk"
    if runtime < RUNTIME_SHORT_MAX:
        return "short"
    elif runtime <= RUNTIME_MEDIUM_MAX:
        return "medium"
    else:
        return "long"


def get_rating_tier(rt: float | None, tmdb_vote: float | None) -> str:
    if rt is not None and not (isinstance(rt, float) and pd.isna(rt)):
        if rt >= RATING_TIER_ACCLAIMED:
            return "acclaimed"
        elif rt >= RATING_TIER_SOLID:
            return "solid"
        elif rt >= RATING_TIER_MIXED:
            return "mixed"
        else:
            return "hidden_gem"
    if tmdb_vote is not None and not (isinstance(tmdb_vote, float) and pd.isna(tmdb_vote)):
        if tmdb_vote >= 7.0:
            return "hidden_gem"
    return "unknown"


def get_popularity_tier(tmdb_votes: int | float | None) -> str:
    if tmdb_votes is None or (isinstance(tmdb_votes, float) and pd.isna(tmdb_votes)):
        return "unknown"
    if tmdb_votes > POPULARITY_BLOCKBUSTER:
        return "blockbuster"
    elif tmdb_votes >= POPULARITY_POPULAR:
        return "popular"
    else:
        return "under_the_radar"


def get_rating_tier_label(tier: str) -> str:
    return {
        "acclaimed": "Acclaimed",
        "solid": "Solid",
        "mixed": "Mixed reviews",
        "hidden_gem": "Hidden gem",
        "unknown": "",
    }.get(tier, "")


def get_popularity_tier_label(tier: str) -> str:
    return {
        "blockbuster": "Blockbuster",
        "popular": "Popular",
        "under_the_radar": "Under the radar",
        "unknown": "",
    }.get(tier, "")


def get_diversity_reason(row: pd.Series, used_decades: set, used_genres: set,
                         used_popularity: set, used_rating_tiers: set) -> str:
    """Generate a short human-readable reason for why this pick adds diversity."""
    parts = []
    genre = row.get("Primary Genre", "")
    decade = get_decade(row.get("Release Year"))
    rating_tier = get_rating_tier(row.get("RT (%)"), row.get("TMDB Vote"))
    pop_tier = get_popularity_tier(row.get("TMDB Votes"))

    tier_label = get_rating_tier_label(rating_tier)
    if tier_label and rating_tier not in used_rating_tiers:
        parts.append(tier_label)

    if genre and genre not in used_genres:
        parts.append(genre.lower())
    elif genre:
        parts.append(genre.lower())

    if decade != "unk" and decade not in used_decades:
        parts.append(f"from the {decade}")

    pop_label = get_popularity_tier_label(pop_tier)
    if pop_label and pop_tier not in used_popularity:
        parts.append(pop_label.lower())

    if not parts:
        parts.append(genre.lower() if genre else "diverse pick")

    return " · ".join(parts).capitalize()


def diverse_surprise_sample(pool: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Select n movies maximizing diversity across 6 dimensions with weighted scoring."""
    if len(pool) <= n:
        return pool.copy()

    candidates = pool.sample(frac=1).reset_index(drop=True)

    selected_order: list[int] = []
    selected_set: set[int] = set()
    used_decades: set[str] = set()
    used_prefixes: set[str] = set()
    used_runtime_cats: set[str] = set()
    used_genres: set[str] = set()
    used_rating_tiers: set[str] = set()
    used_popularity_tiers: set[str] = set()

    for _ in range(n):
        best_idx: int | None = None
        best_diversity = -1
        best_attrs: dict = {}

        for idx in candidates.index:
            if idx in selected_set:
                continue

            row = candidates.loc[idx]
            decade = get_decade(row.get("Release Year"))
            prefix = get_title_key(row.get("Title", ""))
            runtime_cat = get_runtime_category(row.get("Runtime (min)"))
            genre = row.get("Primary Genre", "Unknown")
            rating_tier = get_rating_tier(row.get("RT (%)"), row.get("TMDB Vote"))
            pop_tier = get_popularity_tier(row.get("TMDB Votes"))

            diversity = 0
            if genre not in used_genres:
                diversity += DIVERSITY_WEIGHT_GENRE
            if decade not in used_decades:
                diversity += DIVERSITY_WEIGHT_DEFAULT
            if runtime_cat not in used_runtime_cats:
                diversity += DIVERSITY_WEIGHT_DEFAULT
            if rating_tier not in used_rating_tiers:
                diversity += DIVERSITY_WEIGHT_DEFAULT
            if pop_tier not in used_popularity_tiers:
                diversity += DIVERSITY_WEIGHT_DEFAULT
            if prefix not in used_prefixes:
                diversity += DIVERSITY_WEIGHT_DEFAULT

            if diversity > best_diversity:
                best_diversity = diversity
                best_idx = idx
                best_attrs = {
                    "decade": decade, "prefix": prefix, "runtime_cat": runtime_cat,
                    "genre": genre, "rating_tier": rating_tier, "pop_tier": pop_tier,
                }

        if best_idx is not None:
            selected_order.append(best_idx)
            selected_set.add(best_idx)
            used_decades.add(best_attrs["decade"])
            used_prefixes.add(best_attrs["prefix"])
            used_runtime_cats.add(best_attrs["runtime_cat"])
            used_genres.add(best_attrs["genre"])
            used_rating_tiers.add(best_attrs["rating_tier"])
            used_popularity_tiers.add(best_attrs["pop_tier"])
        else:
            remaining = [i for i in candidates.index if i not in selected_set]
            if remaining:
                selected_order.append(remaining[0])
                selected_set.add(remaining[0])

    return candidates.loc[selected_order].reset_index(drop=True)


# ------------------------------------------------------------------
# Session state initialization
# ------------------------------------------------------------------
if "seen_movies" not in st.session_state:
    st.session_state.seen_movies = load_seen()
if "surprise_reshuffle" not in st.session_state:
    st.session_state.surprise_reshuffle = False
if "omdb_lookups_this_run" not in st.session_state:
    st.session_state.omdb_lookups_this_run = 0
if "omdb_rate_limited" not in st.session_state:
    st.session_state.omdb_rate_limited = False

st.session_state.omdb_lookups_this_run = 0
st.session_state.omdb_rate_limited = False

# ------------------------------------------------------------------
# Load environment variables
# ------------------------------------------------------------------
load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
OMDB_API_KEY = os.getenv("OMDB_API_KEY", "")

# ------------------------------------------------------------------
# Streamlit setup
# ------------------------------------------------------------------
st.set_page_config(page_title="🎬 Movie RT Dashboard", layout="wide")
st.title("🎬 Movie Recommendation Dashboard")

if not TMDB_API_KEY or not OMDB_API_KEY:
    st.error("❌ API keys not detected. Check your .env file in the project root.")
    st.stop()


# ------------------------------------------------------------------
# Cached helpers
# ------------------------------------------------------------------
@st.cache_data(ttl=24 * 3600)
def cached_genres():
    return tmdb_get_genres(TMDB_API_KEY)


@st.cache_data(ttl=24 * 3600)
def cached_discover(genre_ids_key: str, runtime_min, runtime_max, year_min, year_max, page: int):
    genre_ids = [int(g) for g in genre_ids_key.split(",") if g] if genre_ids_key else []
    return tmdb_discover_movies(TMDB_API_KEY, genre_ids, runtime_min, runtime_max, year_min, year_max, page)


@st.cache_data(ttl=24 * 3600)
def cached_tmdb_details(movie_id: int):
    return tmdb_movie_details(TMDB_API_KEY, movie_id)


def tracked_omdb_lookup(imdb_id: str) -> dict | None:
    st.session_state.omdb_lookups_this_run += 1
    result = omdb_lookup_by_imdb_id(OMDB_API_KEY, imdb_id)
    if result and result.get("_rate_limited"):
        st.session_state.omdb_rate_limited = True
        return None
    return result


# ------------------------------------------------------------------
# UI Helper Functions
# ------------------------------------------------------------------
def render_decision_card(
    selected_genres: list[str],
    runtime_bucket: str,
    year_min: int,
    year_max: int,
    min_rt: int,
    require_rt: bool,
    hide_seen: bool,
    surprise_enabled: bool,
    surprise_bias: str,
) -> None:
    with st.container():
        st.markdown("#### 🎯 Current Search")

        cols = st.columns([2, 2, 2, 1])

        with cols[0]:
            genres_text = ", ".join(selected_genres) if selected_genres else "Any"
            st.markdown(f"**Genres:** {genres_text}")
            st.markdown(f"**Runtime:** {runtime_bucket}")

        with cols[1]:
            st.markdown(f"**Years:** {year_min}–{year_max}")
            rt_mode = "Required" if require_rt else "Preferred"
            st.markdown(f"**RT ≥ {min_rt}%** ({rt_mode})")

        with cols[2]:
            seen_status = "Hidden" if hide_seen else "Shown"
            st.markdown(f"**Seen movies:** {seen_status}")
            if surprise_enabled:
                st.markdown(f"**Surprise:** {surprise_bias}")
            else:
                st.markdown("**Surprise:** Off")

        with cols[3]:
            seen_count = len(st.session_state.seen_movies)
            st.metric("Seen", seen_count)


def render_omdb_status(total_movies: int, rt_present: int, imdb_present: int) -> None:
    cache_size = get_cache_size()
    lookups = st.session_state.omdb_lookups_this_run

    if st.session_state.omdb_rate_limited:
        st.warning(
            "⚠️ **OMDb quota reached** — RT/IMDb fields may be missing until daily reset. "
            "Cached data is still available."
        )

    rt_coverage = rt_present / total_movies if total_movies > 0 else 1
    if rt_coverage < 0.5 and not st.session_state.omdb_rate_limited:
        st.info(
            f"ℹ️ Only {rt_present}/{total_movies} movies have RT scores. "
            "Some titles may lack OMDb data."
        )

    st.caption(
        f"OMDb cache: {cache_size:,} entries · "
        f"Lookups this run: {lookups} · "
        f"Coverage — RT: {rt_present}/{total_movies}, IMDb: {imdb_present}/{total_movies}"
    )


def render_rank_explanation(row: pd.Series) -> None:
    rt = row.get("RT (%)")
    tmdb_vote = row.get("TMDB Vote")
    tmdb_votes = row.get("TMDB Votes", 0)
    imdb_rating = row.get("IMDb Rating")
    imdb_votes = row.get("IMDb Votes")
    rank_score_val = row.get("Rank Score", 0)

    with st.expander("🔍 Why this rank?"):
        rt_str = f"{int(rt)}%" if pd.notna(rt) else "N/A"
        tmdb_str = f"{tmdb_vote}/10 ({int(tmdb_votes):,} votes)" if pd.notna(tmdb_vote) else "N/A"

        if pd.notna(imdb_rating) and imdb_rating is not None:
            imdb_votes_int = int(imdb_votes) if pd.notna(imdb_votes) and imdb_votes else 0
            imdb_str = f"{imdb_rating}/10 ({imdb_votes_int:,} votes)"
        else:
            imdb_str = "N/A"

        st.markdown(f"**Inputs:** RT: {rt_str} · TMDB: {tmdb_str} · IMDb: {imdb_str}")

        if pd.notna(rt):
            st.markdown(
                f"**Formula:** {W_RT:.0%} RT (critic) + {W_AUDIENCE:.0%} audience composite + {W_EVIDENCE:.0%} evidence bonus"
            )
            st.caption(
                "Audience = weighted avg of TMDB & IMDb (by vote confidence). "
                "Evidence = how much voting data exists."
            )
        else:
            st.markdown(
                f"**Formula:** {W_AUDIENCE_NO_RT:.0%} audience composite + {W_EVIDENCE_NO_RT:.0%} evidence bonus *(no RT available)*"
            )
            st.caption(
                "Without RT, ranking relies more heavily on TMDB/IMDb audience scores."
            )

        st.markdown(f"**Final Score:** {rank_score_val:.4f}")


def render_movie_card(
    row: pd.Series,
    prefix_key: str,
    show_rank_explanation: bool = False,
    is_wildcard: bool = False,
    genre: str = "",
    diversity_reason: str = "",
) -> None:
    title = row["Title"]
    rt = row["RT (%)"]
    runtime = int(row["Runtime (min)"])
    tmdb_vote = row["TMDB Vote"]
    tmdb_votes = int(row["TMDB Votes"]) if pd.notna(row["TMDB Votes"]) else 0
    overview = row.get("Overview", "")
    release_year = row.get("Release Year", "")

    imdb_rating = row.get("IMDb Rating", None)
    imdb_votes = row.get("IMDb Votes", None)

    rt_text = f"{int(rt)}%" if pd.notna(rt) else "N/A"
    imdb_text = f"{imdb_rating:.1f}/10" if isinstance(imdb_rating, (int, float)) else "N/A"

    if imdb_votes is None or (isinstance(imdb_votes, float) and pd.isna(imdb_votes)):
        imdb_votes_text = "N/A"
    else:
        try:
            imdb_votes_text = f"{int(imdb_votes):,}"
        except Exception:
            imdb_votes_text = "N/A"

    wildcard_badge = " 🃏" if is_wildcard else ""
    genre_badge = f" `{genre}`" if genre else ""

    st.markdown(f"**{title}**{wildcard_badge}{genre_badge}")

    st.markdown(
        f"RT: **{rt_text}** · IMDb: **{imdb_text}** ({imdb_votes_text} votes) · "
        f"TMDB: **{tmdb_vote}** ({tmdb_votes:,} votes)"
    )
    st.markdown(f"{release_year} · {runtime} min")

    if diversity_reason:
        st.caption(f"*{diversity_reason}*")
    elif overview:
        st.caption(overview)

    if show_rank_explanation:
        render_rank_explanation(row)

    seen_key = f"seen::{prefix_key}::{title}"
    is_seen = st.checkbox("Seen", value=(title in st.session_state.seen_movies), key=seen_key)

    if is_seen and title not in st.session_state.seen_movies:
        st.session_state.seen_movies.add(title)
        save_seen(st.session_state.seen_movies)
    elif not is_seen and title in st.session_state.seen_movies:
        st.session_state.seen_movies.discard(title)
        save_seen(st.session_state.seen_movies)

    st.divider()


# ------------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------------
genres_map = cached_genres()
if not genres_map:
    st.warning("Could not load genres from TMDB. Check your API key or try again later.")
genre_names = sorted(genres_map.keys())

with st.sidebar:
    st.header("⚡ Mode")
    view_mode = st.radio(
        "Choose your mode",
        ["Decide Now (90s)", "Browse & Tweak", "Surprise Me"],
        index=0,
        help="Decide Now: Just recommendations. Browse: Full data table. Surprise Me: Diverse discovery picks.",
    )

    surprise_enabled = view_mode == "Surprise Me"

    # Surprise Me settings — only shown in Surprise Me mode
    if surprise_enabled:
        st.divider()
        st.subheader("Surprise settings")
        num_surprise_picks = st.slider("Number of picks", 5, 15, DEFAULT_SURPRISE_PICKS)
        surprise_bias = st.selectbox(
            "Quality bias", ["Pure random", "Prefer higher-ranked"], index=1
        )
        surprise_require_rt = st.checkbox("Require RT score", value=False)
        surprise_exclude_seen = st.checkbox("Exclude seen movies", value=True)

        if st.button("🔀 Reshuffle"):
            st.session_state.surprise_reshuffle = True
            st.rerun()
    else:
        num_surprise_picks = DEFAULT_SURPRISE_PICKS
        surprise_bias = "Prefer higher-ranked"
        surprise_require_rt = False
        surprise_exclude_seen = True

    st.divider()
    st.header("Filters")

    selected_genres = st.multiselect(
        "Genres",
        genre_names,
        default=[g for g in ["Science Fiction", "Fantasy"] if g in genre_names],
    )

    min_rt = st.slider("Minimum Rotten Tomatoes score", 0, 100, 80, 1)

    runtime_bucket = st.selectbox(
        "Runtime", ["< 2 hours", "2–3 hours", "> 3 hours"], index=1
    )

    year_min, year_max = st.slider("Release year range", 1950, CURRENT_YEAR + 1, (2000, CURRENT_YEAR + 1))

    max_pages = st.slider("Search depth (TMDB pages)", 1, 5, 3)

    require_rt = st.checkbox("Require RT score (strict)", value=False)
    st.caption("If strict is off, movies missing RT can appear (ranked lower).")

    st.divider()
    st.header("Watch history")
    hide_seen = st.checkbox("Hide movies I've seen", value=False)

    if st.button("💾 Save"):
        if save_seen(st.session_state.seen_movies):
            st.success("Saved!")
        else:
            st.error("Failed to save watch history.")

genre_ids = [genres_map[g] for g in selected_genres if g in genres_map]
genre_id_to_name = {v: k for k, v in genres_map.items()}
rmin, rmax = runtime_bucket_to_bounds(runtime_bucket)

# ------------------------------------------------------------------
# Determine page depth (adaptive based on Surprise mode)
# ------------------------------------------------------------------
core_pages = max_pages
if surprise_enabled:
    total_pages = min(max_pages + SURPRISE_EXTRA_PAGES, MAX_TOTAL_PAGES)
else:
    total_pages = max_pages

# ------------------------------------------------------------------
# Main logic
# ------------------------------------------------------------------
try:
    render_decision_card(
        selected_genres=selected_genres,
        runtime_bucket=runtime_bucket,
        year_min=year_min,
        year_max=year_max,
        min_rt=min_rt,
        require_rt=require_rt,
        hide_seen=hide_seen,
        surprise_enabled=surprise_enabled,
        surprise_bias=surprise_bias,
    )

    st.divider()

    # 1) Discover movies from TMDB (more pages when Surprise is enabled)
    genre_ids_key = ",".join(map(str, genre_ids))
    candidates = []
    for page in range(1, total_pages + 1):
        data = cached_discover(genre_ids_key, rmin, rmax, year_min, year_max, page)
        results = data.get("results", [])
        for m in results:
            m["_source_page"] = page
        candidates.extend(results)

    if not candidates:
        st.info("No movies returned from TMDB. Try loosening filters.")
        st.stop()

    # 2) Enrich candidates (OMDb only for core pages to limit API usage)
    rows = []
    wildcard_rows = []
    progress = st.progress(0, text="Fetching movie details...")
    total = len(candidates)

    for i, m in enumerate(candidates):
        progress.progress((i + 1) / total, text=f"Processing {i+1}/{total}...")

        movie_id = m.get("id")
        title = m.get("title") or ""
        tmdb_vote = m.get("vote_average")
        tmdb_votes = m.get("vote_count")
        release_date = m.get("release_date") or ""
        source_page = m.get("_source_page", 1)
        is_core = source_page <= core_pages

        details = cached_tmdb_details(movie_id)
        runtime = details.get("runtime")
        imdb_id = details.get("imdb_id")

        if runtime is None:
            continue
        if rmin is not None and runtime < rmin:
            continue
        if rmax is not None and runtime > rmax:
            continue

        rt = None
        imdb_rating = None
        imdb_votes = None

        # Only fetch OMDb for core pages to avoid increasing API usage
        if imdb_id and is_core:
            omdb = tracked_omdb_lookup(imdb_id)
            if omdb:
                rt = extract_rotten_tomatoes_score(omdb)
                imdb_rating, imdb_votes = extract_imdb_score(omdb)

        score = rank_score(
            rt=rt,
            tmdb_vote=tmdb_vote,
            tmdb_votes=tmdb_votes,
            imdb_rating=imdb_rating,
            imdb_votes=imdb_votes,
        )

        genre_names_list = [genre_id_to_name[gid] for gid in m.get("genre_ids", []) if gid in genre_id_to_name]
        primary_genre = genre_names_list[0] if genre_names_list else "Unknown"

        movie_row = {
            "Title": title,
            "Runtime (min)": runtime,
            "RT (%)": rt,
            "IMDb Rating": imdb_rating,
            "IMDb Votes": imdb_votes,
            "TMDB Vote": tmdb_vote,
            "TMDB Votes": tmdb_votes,
            "Rank Score": round(score, 4),
            "Overview": (m.get("overview") or "")[:MAX_OVERVIEW_LENGTH],
            "Release Year": release_date[:4] if len(release_date) >= 4 else "",
            "Genres": ", ".join(genre_names_list[:3]),
            "Primary Genre": primary_genre,
            "_is_core": is_core,
            "_imdb_id": imdb_id,
        }

        # Apply RT filtering for core pages
        if is_core:
            passes_strict = True
            if require_rt and (rt is None or rt < min_rt):
                passes_strict = False
            if rt is not None and rt < min_rt:
                passes_strict = False

            if passes_strict:
                rows.append(movie_row)
            else:
                # Near-miss for wildcard
                is_near_miss = False
                if rt is not None and rt >= max(min_rt - NEAR_MISS_RT_OFFSET, NEAR_MISS_RT_FLOOR):
                    is_near_miss = True
                elif rt is None and tmdb_vote is not None and tmdb_vote >= NEAR_MISS_TMDB_FLOOR:
                    is_near_miss = True
                if is_near_miss:
                    wildcard_rows.append(movie_row)
        else:
            # Non-core pages: include in Surprise pool with looser criteria
            # Must have decent TMDB score since no RT data
            if tmdb_vote is not None and tmdb_vote >= SURPRISE_TMDB_FLOOR:
                rows.append(movie_row)

    progress.empty()
    flush_cache()

    if not rows:
        st.info("No movies matched your filters. Try lowering RT or widening genres.")
        st.stop()

    # Build DataFrames
    df_all = pd.DataFrame(rows)
    df_all = df_all.sort_values(
        ["Rank Score", "RT (%)", "TMDB Vote", "TMDB Votes"], ascending=False
    ).reset_index(drop=True)

    # Core DataFrame for Top 5 (full OMDb data)
    df_core = df_all[df_all["_is_core"] == True].copy().reset_index(drop=True)

    # Wildcard DataFrame
    wildcard_df = pd.DataFrame(wildcard_rows) if wildcard_rows else pd.DataFrame()
    if not wildcard_df.empty:
        wildcard_df = wildcard_df.sort_values(
            ["Rank Score", "TMDB Vote"], ascending=False
        ).reset_index(drop=True)

    # Apply "hide seen" filter
    if hide_seen:
        df_all = df_all[~df_all["Title"].isin(st.session_state.seen_movies)].reset_index(drop=True)
        df_core = df_core[~df_core["Title"].isin(st.session_state.seen_movies)].reset_index(drop=True)
        if not wildcard_df.empty:
            wildcard_df = wildcard_df[
                ~wildcard_df["Title"].isin(st.session_state.seen_movies)
            ].reset_index(drop=True)

    # Coverage stats (from core only, since that's where OMDb was fetched)
    rt_present = int(df_core["RT (%)"].notna().sum()) if "RT (%)" in df_core.columns else 0
    imdb_present = (
        int(df_core["IMDb Rating"].notna().sum()) if "IMDb Rating" in df_core.columns else 0
    )

    render_omdb_status(len(df_core), rt_present, imdb_present)

    # ------------------------------------------------------------------
    # Surprise selection with expanded pool
    # ------------------------------------------------------------------
    surprise_df = None
    wildcard_picks_df = None
    surprise_pool_size = 0
    surprise_diversity_reasons: dict[str, str] = {}

    if surprise_enabled:
        pool = df_all.copy()

        # Apply the same RT filters as other modes
        if surprise_require_rt or require_rt:
            pool = pool[pool["RT (%)"].notna()].reset_index(drop=True)
        # Exclude movies with RT below min_rt (keep those without RT — they'll be enriched later)
        has_rt = pool["RT (%)"].notna()
        pool = pool[~has_rt | (pool["RT (%)"] >= min_rt)].reset_index(drop=True)

        surprise_pool_size = len(pool)

        n_wildcards = max(1, num_surprise_picks // 5)
        n_strict = num_surprise_picks - n_wildcards

        need_resample = (
            "surprise_picks" not in st.session_state
            or st.session_state.get("surprise_reshuffle", False)
        )

        if need_resample and not pool.empty:
            if surprise_bias == "Pure random":
                sample_pool = pool
            else:
                top_n = min(80, len(pool))
                sample_pool = pool.head(top_n)

            if len(sample_pool) <= n_strict:
                strict_picks = sample_pool["Title"].tolist()
            else:
                diverse_sample = diverse_surprise_sample(sample_pool, n=n_strict)
                strict_picks = diverse_sample["Title"].tolist()

            # Expanded wildcard pool: non-core with TMDB >= 6.0, or near-miss core movies
            wc_pool_parts = []
            non_core_wc = df_all[
                (df_all["_is_core"] == False)
                & (df_all["TMDB Vote"] >= WILDCARD_TMDB_FLOOR)
                & (~df_all["Title"].isin(strict_picks))
            ]
            if not non_core_wc.empty:
                wc_pool_parts.append(non_core_wc)
            if not wildcard_df.empty:
                wc_near = wildcard_df[~wildcard_df["Title"].isin(strict_picks)]
                if not wc_near.empty:
                    wc_pool_parts.append(wc_near)

            wc_pool = pd.concat(wc_pool_parts).drop_duplicates(subset="Title") if wc_pool_parts else pd.DataFrame()
            if surprise_exclude_seen and not wc_pool.empty:
                wc_pool = wc_pool[~wc_pool["Title"].isin(st.session_state.seen_movies)]

            # Pick wildcards from different genres than main picks if possible
            wildcard_titles: list[str] = []
            main_genres = set()
            for t in strict_picks:
                match = pool[pool["Title"] == t]
                if not match.empty:
                    main_genres.add(match.iloc[0].get("Primary Genre", ""))

            for _ in range(n_wildcards):
                if wc_pool.empty:
                    break
                diff_genre = wc_pool[~wc_pool["Primary Genre"].isin(main_genres)]
                pick_from = diff_genre if not diff_genre.empty else wc_pool
                chosen = pick_from.sample(n=1).iloc[0]
                wildcard_titles.append(chosen["Title"])
                main_genres.add(chosen.get("Primary Genre", ""))
                wc_pool = wc_pool[wc_pool["Title"] != chosen["Title"]]

            st.session_state.surprise_picks = strict_picks
            st.session_state.surprise_wildcards = wildcard_titles
            st.session_state.surprise_reshuffle = False

        stored_picks = st.session_state.get("surprise_picks", [])
        stored_wildcards = st.session_state.get("surprise_wildcards", [])

        if surprise_exclude_seen:
            display_picks = [t for t in stored_picks if t not in st.session_state.seen_movies]
            display_wildcards = [t for t in stored_wildcards if t not in st.session_state.seen_movies]
        else:
            display_picks = stored_picks
            display_wildcards = stored_wildcards

        if display_picks:
            surprise_df = pool[pool["Title"].isin(display_picks)].reset_index(drop=True)
        else:
            surprise_df = pool.head(0)

        # Build wildcard picks df from all available data
        all_movie_data = pd.concat([df_all, wildcard_df]).drop_duplicates(subset="Title") if not wildcard_df.empty else df_all
        if display_wildcards:
            wildcard_picks_df = all_movie_data[all_movie_data["Title"].isin(display_wildcards)].reset_index(drop=True)
        else:
            wildcard_picks_df = pd.DataFrame()

        # Enrich surprise picks with OMDb (second pass for non-core movies)
        all_surprise_titles = display_picks + display_wildcards
        for title in all_surprise_titles:
            matches = df_all[df_all["Title"] == title]
            if matches.empty:
                matches = wildcard_df[wildcard_df["Title"] == title] if not wildcard_df.empty else pd.DataFrame()
            if matches.empty:
                continue
            row = matches.iloc[0]
            if pd.isna(row.get("RT (%)")) and row.get("_imdb_id"):
                omdb = tracked_omdb_lookup(row["_imdb_id"])
                if omdb:
                    rt_val = extract_rotten_tomatoes_score(omdb)
                    imdb_r, imdb_v = extract_imdb_score(omdb)
                    idx_mask = df_all["Title"] == title
                    if idx_mask.any():
                        df_all.loc[idx_mask, "RT (%)"] = rt_val
                        df_all.loc[idx_mask, "IMDb Rating"] = imdb_r
                        df_all.loc[idx_mask, "IMDb Votes"] = imdb_v
                        new_score = rank_score(rt=rt_val, tmdb_vote=row["TMDB Vote"],
                                               tmdb_votes=row["TMDB Votes"],
                                               imdb_rating=imdb_r, imdb_votes=imdb_v)
                        df_all.loc[idx_mask, "Rank Score"] = round(new_score, 4)
        flush_cache()

        # Refresh surprise_df after enrichment
        if display_picks:
            surprise_df = df_all[df_all["Title"].isin(display_picks)].reset_index(drop=True)
        if display_wildcards and not wildcard_picks_df.empty:
            enriched_wc = df_all[df_all["Title"].isin(display_wildcards)]
            if not enriched_wc.empty:
                wildcard_picks_df = enriched_wc.reset_index(drop=True)

        # Apply RT filters to enriched surprise picks
        def _passes_rt_filter(row):
            rt = row.get("RT (%)")
            if require_rt and (rt is None or (isinstance(rt, float) and pd.isna(rt))):
                return False
            if rt is not None and not (isinstance(rt, float) and pd.isna(rt)) and rt < min_rt:
                return False
            return True

        if surprise_df is not None and not surprise_df.empty:
            surprise_df = surprise_df[surprise_df.apply(_passes_rt_filter, axis=1)].reset_index(drop=True)
        if wildcard_picks_df is not None and not wildcard_picks_df.empty:
            wildcard_picks_df = wildcard_picks_df[wildcard_picks_df.apply(_passes_rt_filter, axis=1)].reset_index(drop=True)

        # Generate diversity reasons
        used_d: set[str] = set()
        used_g: set[str] = set()
        used_p: set[str] = set()
        used_r: set[str] = set()
        if surprise_df is not None:
            for _, r in surprise_df.iterrows():
                reason = get_diversity_reason(r, used_d, used_g, used_p, used_r)
                surprise_diversity_reasons[r["Title"]] = reason
                used_d.add(get_decade(r.get("Release Year")))
                used_g.add(r.get("Primary Genre", ""))
                used_p.add(get_popularity_tier(r.get("TMDB Votes")))
                used_r.add(get_rating_tier(r.get("RT (%)"), r.get("TMDB Vote")))
        if wildcard_picks_df is not None and not wildcard_picks_df.empty:
            for _, r in wildcard_picks_df.iterrows():
                reason = get_diversity_reason(r, used_d, used_g, used_p, used_r)
                surprise_diversity_reasons[r["Title"]] = reason

    # ------------------------------------------------------------------
    # Layout based on mode
    # ------------------------------------------------------------------
    is_decide_mode = view_mode == "Decide Now (90s)"
    is_surprise_mode = view_mode == "Surprise Me"

    if is_surprise_mode:
        # Dedicated Surprise Me view
        has_picks = surprise_df is not None and not surprise_df.empty
        has_wildcards = wildcard_picks_df is not None and not wildcard_picks_df.empty

        if not has_picks and not has_wildcards:
            msg = "No eligible surprise picks. Try widening filters."
            if surprise_require_rt:
                msg = (
                    "No eligible surprise picks (likely too few titles have RT). "
                    "Try turning off 'Require RT score' or widening filters."
                )
            st.info(msg)
        else:
            st.subheader("🎲 Your Surprise Picks")
            pages_used = total_pages
            st.caption(
                f"Sampling from ~{surprise_pool_size} movies ({pages_used} pages) · "
                f"Diverse by genre, era, popularity, and more"
            )

            if has_picks:
                rows_of_3 = [surprise_df.iloc[i:i+3] for i in range(0, len(surprise_df), 3)]
                for row_chunk in rows_of_3:
                    cols = st.columns(3)
                    for col_idx, (_, r) in enumerate(row_chunk.iterrows()):
                        with cols[col_idx]:
                            render_movie_card(
                                r,
                                prefix_key=f"surprise_{col_idx}_{r['Title'][:10]}",
                                genre=r.get("Primary Genre", ""),
                                diversity_reason=surprise_diversity_reasons.get(r["Title"], ""),
                            )

            if has_wildcards:
                st.markdown("### 🃏 Wildcards")
                wc_cols = st.columns(3)
                for wc_idx, (_, r) in enumerate(wildcard_picks_df.iterrows()):
                    with wc_cols[wc_idx % 3]:
                        render_movie_card(
                            r,
                            prefix_key=f"surprise_wc_{wc_idx}",
                            is_wildcard=True,
                            genre=r.get("Primary Genre", ""),
                            diversity_reason=surprise_diversity_reasons.get(r["Title"], ""),
                        )

    elif is_decide_mode:
        st.subheader("🎬 Your Recommendations")

        st.markdown("### ✅ Top 5 Ranked")
        for idx, r in df_core.head(5).iterrows():
            render_movie_card(r, prefix_key=f"top5_{idx}", show_rank_explanation=True)

    else:
        left, right = st.columns([1, 2], gap="large")

        with left:
            st.subheader("✅ Top 5 recommendations (ranked)")
            for idx, r in df_core.head(5).iterrows():
                render_movie_card(r, prefix_key=f"top5_{idx}", show_rank_explanation=True)

        with right:
            st.subheader(f"All matching results ({len(df_core)})")
            display_df = df_core.drop(columns=["Release Year", "_is_core", "_imdb_id", "Genres", "Primary Genre"], errors="ignore")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

except Exception:
    logger.exception("App crashed with unhandled exception")
    st.error("Something went wrong. Please try adjusting your filters and refreshing the page.")
