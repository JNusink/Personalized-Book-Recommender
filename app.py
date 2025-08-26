from fastapi import FastAPI, HTTPException
import numpy as np
from scipy.sparse import load_npz
import joblib
import boto3
import pandas as pd
from decimal import Decimal
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

model_path = 'model.pkl'
matrix_npz_path = 'user_item_matrix.npz'
index_path = 'user_item_indices.npy'
columns_path = 'user_item_columns.npy'

try:
    model = joblib.load(model_path)
    matrix = load_npz(matrix_npz_path)
    indices = np.load(index_path, allow_pickle=True)
    columns = np.load(columns_path, allow_pickle=True)
    user_item_matrix = pd.DataFrame.sparse.from_spmatrix(matrix, index=indices, columns=columns)
    logger.info("Model and matrix loaded successfully")
except Exception as e:
    logger.error(f"Loading failed: {e}")
    raise HTTPException(status_code=500, detail=f"Model/matrix loading error: {e}")

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('BookRecommendations')


@app.post("/predict")
async def predict(user_id: str):
    try:
        logger.info(f"Received request for user_id: {user_id}")
        start_time = time.time()

        if user_id not in user_item_matrix.index:
            raise KeyError("User not found")
        user_idx = user_item_matrix.index.get_loc(user_id)
        user_factors = model.transform(user_item_matrix.values[[user_idx]])
        predicted_ratings = np.dot(user_factors, model.components_)[0]
        if np.any(np.isnan(predicted_ratings)) or np.any(np.isinf(predicted_ratings)):
            predicted_ratings = np.nan_to_num(predicted_ratings, nan=0.0, posinf=0.0, neginf=0.0)
        top_books = user_item_matrix.columns[np.argsort(predicted_ratings)[-5:]].tolist()

        latency = Decimal(str(time.time() - start_time))
        for book_id in top_books:
            predicted_rating_idx = user_item_matrix.columns.get_loc(book_id)
            predicted_rating_value = predicted_ratings[predicted_rating_idx]
            predicted_rating = Decimal(str(predicted_rating_value))
            table.put_item(Item={
                'user_id': user_id,
                'item_id': book_id,
                'predicted_rating': predicted_rating,
                'timestamp': int(time.time()),
                'latency': latency
            })

        return {"user_id": user_id, "recommended_books": top_books}
    except KeyError:
        raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        logger.error(f"Prediction error for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@app.get("/health")
async def health():
    return {"status": "healthy"}
