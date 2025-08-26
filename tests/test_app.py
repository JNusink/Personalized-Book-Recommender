import requests
import pytest
import os


def test_health_endpoint():
    if os.getenv("CI", "false").lower() == "true":
        pytest.skip("Skipping network test in CI environment")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}
    except requests.ConnectionError:
        pytest.fail("FastAPI server not running locally")


def test_predict_endpoint():
    if os.getenv("CI", "false").lower() == "true":
        pytest.skip("Skipping network test in CI environment")
    try:
        response = requests.post("http://localhost:8000/predict?user_id=AE224GVO7OHTYF26U6ER6BEVIUAQ", timeout=5)
        assert response.status_code == 200
        assert "recommended_books" in response.json()
        assert len(response.json()["recommended_books"]) == 5
    except requests.ConnectionError:
        pytest.fail("FastAPI server not running locally")
