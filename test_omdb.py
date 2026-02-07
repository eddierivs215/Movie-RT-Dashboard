import pytest
from omdb import extract_rotten_tomatoes_score, extract_imdb_score


class TestExtractRottenTomatoesScore:
    def test_valid_rt_score(self):
        payload = {"Ratings": [{"Source": "Rotten Tomatoes", "Value": "85%"}]}
        assert extract_rotten_tomatoes_score(payload) == 85

    def test_no_ratings(self):
        assert extract_rotten_tomatoes_score({}) is None

    def test_empty_ratings(self):
        assert extract_rotten_tomatoes_score({"Ratings": []}) is None

    def test_no_rt_source(self):
        payload = {"Ratings": [{"Source": "Internet Movie Database", "Value": "8.0/10"}]}
        assert extract_rotten_tomatoes_score(payload) is None

    def test_100_percent(self):
        payload = {"Ratings": [{"Source": "Rotten Tomatoes", "Value": "100%"}]}
        assert extract_rotten_tomatoes_score(payload) == 100

    def test_0_percent(self):
        payload = {"Ratings": [{"Source": "Rotten Tomatoes", "Value": "0%"}]}
        assert extract_rotten_tomatoes_score(payload) == 0

    def test_invalid_value(self):
        payload = {"Ratings": [{"Source": "Rotten Tomatoes", "Value": "N/A%"}]}
        assert extract_rotten_tomatoes_score(payload) is None

    def test_none_ratings(self):
        assert extract_rotten_tomatoes_score({"Ratings": None}) is None


class TestExtractImdbScore:
    def test_valid_rating_and_votes(self):
        payload = {"imdbRating": "8.5", "imdbVotes": "1,234,567"}
        rating, votes = extract_imdb_score(payload)
        assert rating == 8.5
        assert votes == 1234567

    def test_na_rating(self):
        payload = {"imdbRating": "N/A", "imdbVotes": "1,000"}
        rating, votes = extract_imdb_score(payload)
        assert rating is None
        assert votes == 1000

    def test_na_votes(self):
        payload = {"imdbRating": "7.0", "imdbVotes": "N/A"}
        rating, votes = extract_imdb_score(payload)
        assert rating == 7.0
        assert votes is None

    def test_empty_payload(self):
        rating, votes = extract_imdb_score({})
        assert rating is None
        assert votes is None

    def test_none_payload(self):
        rating, votes = extract_imdb_score(None)
        assert rating is None
        assert votes is None

    def test_missing_fields(self):
        rating, votes = extract_imdb_score({"Title": "Test"})
        assert rating is None
        assert votes is None
