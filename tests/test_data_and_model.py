"""Sanity checks on the cleaned dataset and the saved model artifacts."""

import os

import joblib
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT, "data", "processed", "repos_clean.csv")
MODEL_DIR = os.path.join(ROOT, "src", "model")

EXPECTED_FEATURES = [
    "stargazers_count", "forks_count", "first_month_commits", "contributors_count",
    "issue_close_ratio", "commits_per_contributor",
    "has_license", "has_wiki", "has_description",
]


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def model():
    return joblib.load(os.path.join(MODEL_DIR, "repo_survival_model.pkl"))


@pytest.fixture(scope="module")
def feature_cols():
    return joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))


def test_label_is_binary_and_not_degenerate(df):
    assert set(df["label"].unique()) == {0, 1}
    assert 0.2 < df["label"].mean() < 0.5  # about 31% active


def test_no_duplicate_repos(df):
    assert df["id"].is_unique


def test_features_have_no_missing_values(df):
    assert df[EXPECTED_FEATURES].isna().sum().sum() == 0


def test_ratios_are_in_range(df):
    assert df["issue_close_ratio"].between(0, 1).all()
    assert (df["commits_per_contributor"] >= 0).all()


def test_saved_features_match_expected(feature_cols):
    assert list(feature_cols) == EXPECTED_FEATURES


def test_age_days_is_not_a_model_feature(feature_cols):
    """age_days leaks the label definition and must stay excluded."""
    assert "age_days" not in feature_cols
    assert "days_since_push" not in feature_cols


def test_model_outputs_valid_probabilities(df, model, feature_cols):
    probabilities = model.predict_proba(df[feature_cols].head(50))
    assert probabilities.shape == (50, 2)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_model_ranks_better_than_chance(df, model, feature_cols):
    """Loose floor (the model was trained on part of this data, so this is a
    smoke test and not an evaluation)."""
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(df["label"], model.predict_proba(df[feature_cols])[:, 1])
    assert auc > 0.75
