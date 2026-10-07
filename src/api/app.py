"""
Flask API that receives a GitHub repo URL, computes its first-month-of-life
features, and returns a survival prediction using the trained model.
"""

import os
import re
import sys

import joblib
import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Allows importing the functions already written in collect_details.py
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from collect_details import (HEADERS, get_contributors_count,  # noqa: E402
                             get_first_month_commits, get_issues_counts)

load_dotenv()

WEB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "web"))

# Flask also serves the static frontend, so the whole app is a single service
# (same origin for the page and the API, which is what the deployment uses).
app = Flask(__name__, static_folder=WEB_DIR, static_url_path="")
CORS(app)  # still needed for local development with the frontend on another port

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "repo_survival_model.pkl")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "feature_cols.pkl")

model = joblib.load(MODEL_PATH)
feature_cols = joblib.load(FEATURES_PATH)

REQUEST_TIMEOUT = 15  # seconds

# Accepts, among others:
#   https://github.com/owner/repo
#   https://github.com/owner/repo/
#   https://github.com/owner/repo.git
#   https://github.com/owner/repo/tree/main/src
#   github.com/owner/repo
#   git@github.com:owner/repo.git
GITHUB_URL_PATTERN = re.compile(
    r"^(?:git@|https?://)?(?:www\.)?github\.com[/:]"
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)/"
    r"(?P<repo>[A-Za-z0-9._-]+?)"
    r"(?:\.git)?(?:[/?#].*)?$"
)


class GitHubError(Exception):
    """Raised when GitHub answers with something other than a usable repo."""

    def __init__(self, message, status_code):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def parse_github_url(url):
    """Extract (owner, repo) from a GitHub URL, or (None, None) if invalid."""
    match = GITHUB_URL_PATTERN.match((url or "").strip())
    if not match:
        return None, None
    return match.group("owner"), match.group("repo")


def get_repo_metadata(owner, repo):
    """Fetch basic repo metadata (stars, forks, license...).

    Raises GitHubError with an accurate message, so a rate limit or a network
    problem is not reported as "repo not found".
    """
    url = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    except requests.RequestException:
        raise GitHubError("Could not reach the GitHub API", 502)

    if response.status_code == 200:
        return response.json()
    if response.status_code == 404:
        raise GitHubError("Repo not found on GitHub (it may be private)", 404)
    if response.status_code in (403, 429):
        raise GitHubError("GitHub API rate limit reached, try again later", 503)
    if response.status_code == 401:
        raise GitHubError("GitHub token is invalid or missing on the server", 500)
    raise GitHubError(f"Unexpected GitHub response ({response.status_code})", 502)


@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    repo_url = data.get("repo_url", "")

    owner, repo = parse_github_url(repo_url)
    if not owner or not repo:
        return jsonify({"error": "Invalid GitHub URL"}), 400

    try:
        metadata = get_repo_metadata(owner, repo)

        # Compute the same features as in collect_details.py
        first_month_commits = get_first_month_commits(owner, repo, metadata["created_at"])
        contributors_count = get_contributors_count(owner, repo)
        open_issues, closed_issues = get_issues_counts(owner, repo)
    except GitHubError as error:
        return jsonify({"error": error.message}), error.status_code
    except requests.RequestException:
        return jsonify({"error": "Could not reach the GitHub API"}), 502
    except Exception:
        app.logger.exception("Feature computation failed for %s/%s", owner, repo)
        return jsonify({"error": "Could not compute the features for this repo"}), 502

    total_issues = open_issues + closed_issues
    issue_close_ratio = closed_issues / total_issues if total_issues > 0 else 0
    commits_per_contributor = first_month_commits / contributors_count if contributors_count > 0 else 0

    # Build the feature vector in the SAME order as feature_cols
    features = {
        "stargazers_count": metadata["stargazers_count"],
        "forks_count": metadata["forks_count"],
        "first_month_commits": first_month_commits,
        "contributors_count": contributors_count,
        "issue_close_ratio": issue_close_ratio,
        "commits_per_contributor": commits_per_contributor,
        "has_license": metadata["license"] is not None,
        "has_wiki": metadata["has_wiki"],
        "has_description": metadata["description"] is not None,
    }

    X = pd.DataFrame([features])[feature_cols]
    probability = model.predict_proba(X)[0][1]

    return jsonify({
        "owner": owner,
        "repo": repo,
        "survival_probability": round(float(probability), 3),
        "prediction": "Active" if probability >= 0.5 else "Abandoned",
        "features_used": features,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
