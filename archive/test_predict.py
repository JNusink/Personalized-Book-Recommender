import requests

user_id = "AE224GVO7OHTYF26U6ER6BEVIUAQ"  # Use a known user ID
response = requests.post(f"http://localhost:8000/predict?user_id={user_id}")
print(response.status_code, response.json())
