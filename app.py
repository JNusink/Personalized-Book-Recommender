from fastapi import FastAPI, HTTPException
import mlflow.sklearn
import numpy as np
import joblib
import boto3
import pandas as pd
from decimal import Decimal, getcontext, InvalidOperation
import logging
import time

getcontext().prec = 15  # Increased precision

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
try:
    model = mlflow.sklearn.load_model("models:/BookRecommenderModel/18")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    raise

user_item_matrix = joblib.load('user_item_matrix.pkl')
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('BookRecommendations')

@app.post("/predict")
async def predict(user_id: str):
    try:
        logger.info(f"Received request for user_id: {user_id}")
        start_time = time.time()
        logger.debug(f"Start time (Unix): {start_time}")
        user_idx = user_item_matrix.index.get_loc(user_id)
        logger.info(f"Found user_idx: {user_idx}")
        user_factors = model.transform(user_item_matrix.values[[user_idx]])
        predicted_ratings = np.dot(user_factors, model.components_)[0]
        logger.debug(f"Predicted ratings array: {predicted_ratings}")
        if np.any(np.isnan(predicted_ratings)) or np.any(np.isinf(predicted_ratings)):
            logger.warning(f"NaN or Inf detected in predicted_ratings for {user_id}")
            predicted_ratings = np.nan_to_num(predicted_ratings, nan=0.0, posinf=0.0, neginf=0.0)
        top_books = user_item_matrix.columns[np.argsort(predicted_ratings)[-5:]].tolist()
        logger.info(f"Top books for {user_id}: {top_books}")

        for i, book_id in enumerate(top_books):
            current_time = time.time()
            logger.debug(f"Current time (Unix) for {book_id}: {current_time}")
            latency = Decimal.from_float(float(current_time - start_time)).to_eng_string()  # Removed quantize
            predicted_rating_idx = user_item_matrix.columns.get_loc(book_id)
            predicted_rating_value = predicted_ratings[predicted_rating_idx]
            logger.debug(f"Predicted rating value for {book_id} (index {predicted_rating_idx}): {predicted_rating_value}")
            try:
                predicted_rating = Decimal.from_float(float(predicted_rating_value)).to_eng_string()
            except InvalidOperation as e:
                logger.error(f"InvalidOperation for {book_id}: {e}, value: {predicted_rating_value}")
                predicted_rating = Decimal('0.0').to_eng_string()  # Use Decimal object
            logger.debug(f"Converted predicted_rating for {book_id}: {predicted_rating}")
            table.put_item(Item={
                'user_id': user_id,
                'item_id': str(book_id),
                'predicted_rating': predicted_rating,
                'timestamp': int(current_time),
                'latency': latency
            })
            logger.info(f"Logged {book_id} for {user_id}")
        return {"user_id": user_id, "recommended_books": top_books}
    except KeyError:
        logger.error(f"User {user_id} not found in matrix")
        raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.get("/health")
async def health():
    return {"status": "healthy"}