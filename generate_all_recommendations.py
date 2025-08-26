import requests
import numpy as np
import json

# Load all user IDs from the saved numpy file
user_ids = np.load('user_item_indices.npy', allow_pickle=True)

# Initialize a dictionary to store all recommendations
all_recommendations = {}

# Query the FastAPI endpoint for each user ID
for user_id in user_ids:
    try:
        response = requests.post(f"http://localhost:8000/predict?user_id={user_id}")
        if response.status_code == 200:
            all_recommendations[user_id] = response.json()["recommended_books"]
        else:
            all_recommendations[user_id] = f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        all_recommendations[user_id] = f"Error: {str(e)}"

# Save to a JSON file
with open('all_recommendations.json', 'w') as f:
    json.dump(all_recommendations, f, indent=4)

print("Recommendations for all users saved to all_recommendations.json")
