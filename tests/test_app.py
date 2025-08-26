import requests


def test_health_endpoint():
    response = requests.get("http://localhost:8000/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_endpoint():
    response = requests.post("http://localhost:8000/predict?user_id=AE224GVO7OHTYF26U6ER6BEVIUAQ")
    assert response.status_code == 200
    assert "recommended_books" in response.json()
    assert len(response.json()["recommended_books"]) == 5  # Expect 5 recommendations
