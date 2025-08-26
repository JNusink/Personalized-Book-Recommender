import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import os
import warnings
import logging
import boto3
import traceback
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress MLflow warnings
os.environ["MLFLOW_LOGGING_WARNINGS"] = "0"
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow.*")

try:
    # Load data
    logger.info("Loading books_5core_1M.parquet")
    df = pd.read_parquet('books_5core_1M.parquet')

    # Verify aggregation method
    logger.info("Using most recent rating aggregation (un-normalized, threshold 2, 1M dataset)")

    # Ensure ratings are un-normalized (1-5)
    if df['rating'].max() <= 1.0:
        df['rating'] = df['rating'] * 5
    logger.info(f"Rating range: {df['rating'].min()} - {df['rating'].max()}")

    # Aggregate duplicate user_id/parent_asin pairs by most recent rating
    df = df.sort_values('timestamp').groupby(['user_id', 'parent_asin'])[['rating', 'timestamp']].last().reset_index()

    # Create user-item matrix (no normalization)
    user_item_matrix = df.pivot(index='user_id', columns='parent_asin', values='rating').fillna(0)
    sparse_matrix = csr_matrix(user_item_matrix.values)

    # Save sparse matrix and indices
    save_npz('user_item_matrix.npz', sparse_matrix)
    user_ids = user_item_matrix.index.values
    np.save('user_item_indices.npy', user_ids)
    np.save('user_item_columns.npy', user_item_matrix.columns.values)

    # Split into train/test
    train_matrix, test_matrix = train_test_split(sparse_matrix, test_size=0.2, random_state=42)

    # Set experiment
    mlflow.set_experiment("book_recommender")

    with mlflow.start_run():
        # Parameters
        params = {
            "n_components": 1000,
            "sample_size": len(df),
            "n_users": user_item_matrix.shape[0],
            "n_items": user_item_matrix.shape[1],
            "sparsity": 1 - (len(df) / (user_item_matrix.shape[0] * user_item_matrix.shape[1])),
            "data_version": "books_5core_1M",
            "n_iter": 10,
            "aggregation": "most_recent",
            "normalized": False,
            "precision_threshold": 2
        }
        mlflow.log_params(params)

        # Train SVD (Model 18 configuration)
        svd = TruncatedSVD(n_components=1000, random_state=42, n_iter=10)
        user_factors = svd.fit_transform(train_matrix)

        # Save model locally
        save_path = 'model.pkl'
        joblib.dump(svd, save_path)

        # Metrics
        predicted_ratings = np.dot(user_factors, svd.components_)
        train_rmse = np.sqrt(mean_squared_error(train_matrix.toarray(), predicted_ratings))
        test_user_factors = svd.transform(test_matrix)
        test_predicted_ratings = np.dot(test_user_factors, svd.components_)
        test_rmse = np.sqrt(mean_squared_error(test_matrix.toarray(), test_predicted_ratings))

        def precision_at_k(actual, predicted, k=5):
            precisions = []
            for user_idx in range(actual.shape[0]):
                top_k_items = np.argsort(predicted[user_idx])[-k:]
                actual_ratings = actual[user_idx, top_k_items].toarray().flatten()
                precision = np.sum(actual_ratings >= 2) / k
                precisions.append(precision)
            return np.mean(precisions) if precisions else 0.0

        train_precision = precision_at_k(train_matrix, predicted_ratings)
        test_precision = precision_at_k(test_matrix, test_predicted_ratings)

        mlflow.log_metrics({
            "train_rmse": float(train_rmse),
            "test_rmse": float(test_rmse),
            "train_precision_5": float(train_precision),
            "test_precision_5": float(test_precision)
        })

        # Log model with signature
        input_example = user_item_matrix.iloc[[0]].values
        signature = infer_signature(input_example, predicted_ratings[0])
        mlflow.sklearn.log_model(svd, "svd_model", signature=signature)
        mlflow.log_artifact('user_item_matrix.npz')
        mlflow.log_artifact('user_item_indices.npy')
        mlflow.log_artifact('user_item_columns.npy')

        # Register model
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/svd_model"
        mlflow.register_model(model_uri, "BookRecommenderModel")

        # Generate recommendations for all users
        logger.info("Generating recommendations for all users")
        all_user_factors = svd.transform(sparse_matrix)
        all_predicted_ratings = np.dot(all_user_factors, svd.components_)
        all_recommendations = {}
        for idx, user_id in enumerate(user_ids):
            top_items = user_item_matrix.columns[np.argsort(all_predicted_ratings[idx])[-5:]].tolist()
            all_recommendations[user_id] = top_items

        # Save all recommendations
        with open('all_recommendations.json', 'w') as f:
            json.dump(all_recommendations, f, indent=4)
        mlflow.log_artifact('all_recommendations.json')

        # Upload to S3 with logging
        if os.path.exists(save_path):
            logger.debug("Uploading to S3")
            try:
                s3 = boto3.client('s3')
                bucket_name = 'my-book-recommender-2025-jtnusink'
                for file in [
                    'model.pkl',
                    'user_item_matrix.npz',
                    'user_item_indices.npy',
                    'user_item_columns.npy',
                    'books_5core_1M.parquet',
                    'all_recommendations.json'
                ]:
                    s3.upload_file(file, bucket_name, f'data/{file}')
                    logger.info(f"Uploaded {file} to s3://{bucket_name}/data/{file}")
            except Exception as e:
                logger.error(f"S3 upload failed: {e}")
                traceback.print_exc()
        else:
            logger.warning("Skipping S3 upload due to missing model.pkl")

        logger.info(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
        logger.info(f"Train Precision@5: {train_precision:.4f}, Test Precision@5: {test_precision:.4f}")

except Exception as e:
    logger.error(f"Error in training process: {e}")
    traceback.print_exc()
    raise
