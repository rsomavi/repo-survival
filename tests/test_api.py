"""Tests for the Flask API. GitHub and the trained model are mocked, so they
run offline, fast, and do not use any API quota."""

import numpy as np
import pytest

import app as api


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class FakeModel:
    """Always predicts a fixed survival probability."""

    def __init__(self, probability):
        self.probability = probability

    def predict_proba(self, X):
        return np.array([[1 - self.probability, self.probability]] * len(X))


FAKE_METADATA = {
    "created_at": "2023-03-01T10:00:00Z",
    "stargazers_count": 120,
    "forks_count": 15,
    "license": {"key": "mit"},
    "has_wiki": True,
    "description": "A test repo",
}


@pytest.fixture
def mock_github(monkeypatch):
    """Replaces every GitHub-dependent function with a deterministic fake."""
    monkeypatch.setattr(api, "get_repo_metadata", lambda owner, repo: FAKE_METADATA)
    monkeypatch.setattr(api, "get_first_month_commits", lambda owner, repo, created: 40)
    monkeypatch.setattr(api, "get_contributors_count", lambda owner, repo: 8)
    monkeypatch.setattr(api, "get_issues_counts", lambda owner, repo: (2, 6))


@pytest.mark.parametrize("url, expected", [
    ("https://github.com/pallets/flask", ("pallets", "flask")),
    ("https://github.com/pallets/flask/", ("pallets", "flask")),
    ("https://github.com/pallets/flask.git", ("pallets", "flask")),
    ("github.com/pallets/flask", ("pallets", "flask")),
    ("https://github.com/pallets/flask/tree/main/src", ("pallets", "flask")),
    ("git@github.com:rsomavi/minidbms.git", ("rsomavi", "minidbms")),
    ("  https://github.com/rsomavi/minidbms.git  ", ("rsomavi", "minidbms")),
    ("https://github.com/owner/repo.with.dots", ("owner", "repo.with.dots")),
])
def test_parse_valid_urls(url, expected):
    assert api.parse_github_url(url) == expected


@pytest.mark.parametrize("url", [
    "", "hola", "https://github.com/pallets", "https://gitlab.com/a/b", None,
])
def test_parse_invalid_urls(url):
    assert api.parse_github_url(url) == (None, None)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_predict_invalid_url_returns_400(client):
    response = client.post("/predict", json={"repo_url": "hola"})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid GitHub URL"


def test_predict_without_body_returns_400(client):
    response = client.post("/predict")
    assert response.status_code == 400


def test_predict_success(client, mock_github, monkeypatch):
    monkeypatch.setattr(api, "model", FakeModel(0.8))
    response = client.post("/predict", json={"repo_url": "https://github.com/a/b.git"})
    body = response.get_json()

    assert response.status_code == 200
    assert body["owner"] == "a" and body["repo"] == "b"
    assert body["survival_probability"] == 0.8
    assert body["prediction"] == "Active"
    assert body["features_used"]["issue_close_ratio"] == pytest.approx(0.75)
    assert body["features_used"]["commits_per_contributor"] == pytest.approx(5.0)
    assert body["features_used"]["has_license"] is True


def test_predict_low_probability_is_abandoned(client, mock_github, monkeypatch):
    monkeypatch.setattr(api, "model", FakeModel(0.2))
    body = client.post("/predict", json={"repo_url": "https://github.com/a/b"}).get_json()
    assert body["prediction"] == "Abandoned"


def test_zero_contributors_and_issues_do_not_divide_by_zero(client, mock_github, monkeypatch):
    monkeypatch.setattr(api, "model", FakeModel(0.5))
    monkeypatch.setattr(api, "get_contributors_count", lambda owner, repo: 0)
    monkeypatch.setattr(api, "get_issues_counts", lambda owner, repo: (0, 0))
    response = client.post("/predict", json={"repo_url": "https://github.com/a/b"})
    features = response.get_json()["features_used"]

    assert response.status_code == 200
    assert features["issue_close_ratio"] == 0
    assert features["commits_per_contributor"] == 0


def test_feature_vector_matches_trained_columns(client, mock_github, monkeypatch):
    """The API must send exactly the columns the model was trained on."""
    monkeypatch.setattr(api, "model", FakeModel(0.5))
    features = client.post("/predict", json={"repo_url": "https://github.com/a/b"}).get_json()["features_used"]
    assert set(api.feature_cols) <= set(features)


@pytest.mark.parametrize("status, expected", [
    (404, 404),   # not found or private
    (403, 503),   # rate limit
    (429, 503),
    (401, 500),   # bad server token
    (500, 502),   # unexpected GitHub error
])
def test_github_status_codes_are_mapped(monkeypatch, status, expected):
    monkeypatch.setattr(api.requests, "get", lambda *a, **k: FakeResponse(status))
    with pytest.raises(api.GitHubError) as error:
        api.get_repo_metadata("a", "b")
    assert error.value.status_code == expected


def test_network_failure_returns_502(monkeypatch):
    def boom(*args, **kwargs):
        raise api.requests.ConnectionError()

    monkeypatch.setattr(api.requests, "get", boom)
    with pytest.raises(api.GitHubError) as error:
        api.get_repo_metadata("a", "b")
    assert error.value.status_code == 502


def test_predict_returns_404_for_missing_repo(client, monkeypatch):
    monkeypatch.setattr(api.requests, "get", lambda *a, **k: FakeResponse(404))
    response = client.post("/predict", json={"repo_url": "https://github.com/a/missing"})
    assert response.status_code == 404
    assert "not found" in response.get_json()["error"].lower()
