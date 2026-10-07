"""Shared pytest setup: makes `src/` importable and exposes a Flask test client."""

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "api"))

import app as api  # noqa: E402


@pytest.fixture
def client():
    api.app.config["TESTING"] = True
    return api.app.test_client()
