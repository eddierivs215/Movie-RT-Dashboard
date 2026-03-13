import pytest
from recommend import rank_score, runtime_bucket_to_bounds, repetition_penalty, discovery_score, _safe_float, _safe_int, _clamp


class TestRuntimeBucketToBounds:
    def test_under_1_hour(self):
        assert runtime_bucket_to_bounds("< 1 hour") == (0, 59)

    def test_1_to_2_hours(self):
        assert runtime_bucket_to_bounds("1–2 hours") == (60, 119)

    def test_over_2_hours(self):
        assert runtime_bucket_to_bounds("> 2 hours") == (120, None)

    def test_unknown_bucket(self):
        assert runtime_bucket_to_bounds("something") == (None, None)

    def test_empty_string(self):
        assert runtime_bucket_to_bounds("") == (None, None)

    def test_none(self):
        assert runtime_bucket_to_bounds(None) == (None, None)


class TestSafeFloat:
    def test_valid_float(self):
        assert _safe_float(3.14) == 3.14

    def test_valid_int(self):
        assert _safe_float(5) == 5.0

    def test_string_float(self):
        assert _safe_float("7.5") == 7.5

    def test_none(self):
        assert _safe_float(None) is None

    def test_nan(self):
        assert _safe_float(float("nan")) is None

    def test_invalid_string(self):
        assert _safe_float("abc") is None


class TestSafeInt:
    def test_valid_int(self):
        assert _safe_int(42) == 42

    def test_float_to_int(self):
        assert _safe_int(7.9) == 7

    def test_string_with_commas(self):
        assert _safe_int("12,345") == 12345

    def test_none(self):
        assert _safe_int(None) is None

    def test_invalid_string(self):
        assert _safe_int("abc") is None


class TestClamp:
    def test_within_range(self):
        assert _clamp(0.5) == 0.5

    def test_below_range(self):
        assert _clamp(-0.5) == 0.0

    def test_above_range(self):
        assert _clamp(1.5) == 1.0

    def test_at_bounds(self):
        assert _clamp(0.0) == 0.0
        assert _clamp(1.0) == 1.0


class TestRankScore:
    def test_all_inputs_present(self):
        score = rank_score(rt=90, tmdb_vote=8.0, tmdb_votes=3000, imdb_rating=8.5, imdb_votes=40000)
        assert 0.0 <= score <= 1.0
        assert score > 0.7  # High-quality movie should score well

    def test_missing_rt(self):
        score = rank_score(rt=None, tmdb_vote=7.0, tmdb_votes=2000, imdb_rating=7.5, imdb_votes=30000)
        assert 0.0 <= score <= 1.0

    def test_missing_all_audience(self):
        score = rank_score(rt=85, tmdb_vote=None, tmdb_votes=None)
        assert 0.0 <= score <= 1.0

    def test_all_none(self):
        score = rank_score(rt=None, tmdb_vote=None, tmdb_votes=None)
        assert score == 0.0

    def test_zero_votes(self):
        score = rank_score(rt=90, tmdb_vote=8.0, tmdb_votes=0, imdb_rating=8.0, imdb_votes=0)
        assert 0.0 <= score <= 1.0

    def test_rt_dominates_when_present(self):
        high_rt = rank_score(rt=95, tmdb_vote=5.0, tmdb_votes=1000)
        low_rt = rank_score(rt=30, tmdb_vote=9.0, tmdb_votes=1000)
        assert high_rt > low_rt

    def test_higher_votes_increase_confidence(self):
        low_votes = rank_score(rt=None, tmdb_vote=7.0, tmdb_votes=100)
        high_votes = rank_score(rt=None, tmdb_vote=7.0, tmdb_votes=5000)
        assert high_votes > low_votes

    def test_score_clamped_to_unit_interval(self):
        score = rank_score(rt=100, tmdb_vote=10.0, tmdb_votes=100000, imdb_rating=10.0, imdb_votes=500000)
        assert score <= 1.0

    def test_perfect_score(self):
        score = rank_score(rt=100, tmdb_vote=10.0, tmdb_votes=10000, imdb_rating=10.0, imdb_votes=100000)
        assert score == 1.0

    def test_only_imdb(self):
        score = rank_score(rt=None, tmdb_vote=None, tmdb_votes=None, imdb_rating=7.0, imdb_votes=20000)
        assert 0.0 < score < 1.0


class TestRepetitionPenalty:
    def test_no_exposure(self):
        assert repetition_penalty(0, None) == 1.0

    def test_shown_once_today(self):
        from datetime import datetime
        today = datetime.now().isoformat()
        penalty = repetition_penalty(1, today)
        # Shown today: recency ≈ 0, so penalty near 0
        assert 0.0 <= penalty < 0.15

    def test_shown_once_old(self):
        # Shown 30 days ago — recency = 1.0, only frequency penalty applies
        penalty = repetition_penalty(1, "2020-01-01T00:00:00")
        assert 0.5 < penalty <= 1.0

    def test_shown_many_times_today(self):
        from datetime import datetime
        today = datetime.now().isoformat()
        penalty = repetition_penalty(20, today)
        assert penalty < 0.1

    def test_shown_many_times_old(self):
        penalty = repetition_penalty(20, "2020-01-01T00:00:00")
        # Frequency penalty for 20 showings with FREQUENCY_WEIGHT=0.3
        # = 1 / (1 + log(21) * 0.3) ≈ 1 / (1 + 0.91) ≈ 0.52
        assert 0.3 < penalty < 0.7

    def test_invalid_date(self):
        # Invalid date string should not crash, uses RECENCY_WINDOW_DAYS fallback
        penalty = repetition_penalty(1, "not-a-date")
        # recency = min(1.0, 14/14) = 1.0; frequency near 1.0 for count=1
        assert 0.0 < penalty <= 1.0

    def test_penalty_between_zero_and_one(self):
        from datetime import datetime
        today = datetime.now().isoformat()
        for count in [0, 1, 5, 20]:
            p = repetition_penalty(count, today if count > 0 else None)
            assert 0.0 <= p <= 1.0


class TestDiscoveryScore:
    def test_below_quality_floor(self):
        assert discovery_score(base_rank_score=0.4, tmdb_votes=500) == 0.0

    def test_above_popularity_cap(self):
        assert discovery_score(base_rank_score=0.8, tmdb_votes=5000) == 0.0

    def test_below_min_votes(self):
        assert discovery_score(base_rank_score=0.8, tmdb_votes=10) == 0.0

    def test_valid_fresh_item(self):
        score = discovery_score(base_rank_score=0.75, tmdb_votes=300, shown_count=0, last_shown_at=None)
        # freshness = 1 + 0.4 = 1.4; obscurity boost since 300 < 2000
        assert score > 0.75

    def test_valid_stale_item(self):
        # Shown recently: freshness close to 1.0
        from datetime import datetime
        score_fresh = discovery_score(0.75, 300, shown_count=0, last_shown_at=None)
        score_stale = discovery_score(0.75, 300, shown_count=3, last_shown_at=datetime.now().isoformat())
        assert score_fresh > score_stale

    def test_lower_votes_higher_score(self):
        # Lower popularity → higher obscurity bonus
        score_low = discovery_score(0.75, 50)
        score_high = discovery_score(0.75, 1800)
        assert score_low > score_high

    def test_none_votes(self):
        # None votes treated as 0, below DISCOVERY_MIN_VOTES → 0.0
        assert discovery_score(base_rank_score=0.8, tmdb_votes=None) == 0.0

    def test_freshness_max_when_never_shown(self):
        score_never = discovery_score(0.7, 200, shown_count=0, last_shown_at=None)
        score_old = discovery_score(0.7, 200, shown_count=1, last_shown_at="2020-01-01T00:00:00")
        # Both get full freshness (never shown = max; old date = max days elapsed)
        assert abs(score_never - score_old) < 0.01
