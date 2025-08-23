import requests

def test_predict_endpoint():
    url = "http://localhost:8000/predict?user_id=AE224GVO7OHTYF26U6ER6BEVIUAQ"
    response = requests.post(url)
    assert response.status_code == 200
    data = response.json()
    assert "user_id" in data and "recommended_books" in data
    print("Predict endpoint test passed!")

def test_health_endpoint():
    url = "http://localhost:8000/health"
    response = requests.get(url)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    print("Health endpoint test passed!")

if __name__ == "__main__":
    test_predict_endpoint()
    test_health_endpoint()
