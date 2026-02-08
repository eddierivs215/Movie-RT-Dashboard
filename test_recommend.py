import pytest
from recommend import rank_score, runtime_bucket_to_bounds, _safe_float, _safe_int, _clamp


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
