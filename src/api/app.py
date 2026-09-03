"""
Flask API that receives a GitHub repo URL, computes its first-month-of-life
features, and returns a survival prediction using the trained model.
"""

import os
import sys
import re
import joblib
import requests
from datetime import datetime
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Allows importing the functions already written in collect_details.py
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from collect_details import get_first_month_commits, get_contributors_count, get_issues_counts, HEADERS

load_dotenv()

app = Flask(__name__)
CORS(app)  # allow the frontend (different port/origin) to call this API

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "repo_survival_model.pkl")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "feature_cols.pkl")

model = joblib.load(MODEL_PATH)
feature_cols = joblib.load(FEATURES_PATH)


def parse_github_url(url):
    """Extract owner/repo from a GitHub URL."""
    match = re.search(r"github\.com/([^/]+)/([^/]+?)/?$", url.strip())
    if not match:
        return None, None
    return match.group(1), match.group(2)


def get_repo_metadata(owner, repo):
    """Fetch basic repo metadata (stars, forks, license...)."""
    url = f"https://api.github.com/repos/{owner}/{repo}"
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        return None

    return response.json()


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    repo_url = data.get("repo_url", "")

    owner, repo = parse_github_url(repo_url)
    if not owner or not repo:
        return jsonify({"error": "Invalid GitHub URL"}), 400

    metadata = get_repo_metadata(owner, repo)
    if metadata is None:
        return jsonify({"error": "Repo not found on GitHub"}), 404

    # Compute the same features as in collect_details.py
    first_month_commits = get_first_month_commits(owner, repo, metadata["created_at"])
    contributors_count = get_contributors_count(owner, repo)
    open_issues, closed_issues = get_issues_counts(owner, repo)

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