import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
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

# Suppress MLflow warnings
os.environ["MLFLOW_LOGGING_WARNINGS"] = "0"
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow.*")
logging.getLogger("mlflow").setLevel(logging.CRITICAL)

# Load data
try:
    df = pd.read_parquet('books_5core_1M.parquet')
except FileNotFoundError:
    print("Error: books_5core_1M.parquet not found. Run preprocess_1M.py first.")
    exit(1)

# Verify aggregation method
print("Using most recent rating aggregation (un-normalized, threshold 2, 1M dataset)")

# Ensure ratings are un-normalized (1-5)
if df['rating'].max() <= 1.0:
    df['rating'] = df['rating'] * 5
print("Rating range:", df['rating'].min(), "-", df['rating'].max())

# Aggregate duplicate user_id/parent_asin pairs by most recent rating
df = df.sort_values('timestamp').groupby(['user_id', 'parent_asin'])[['rating', 'timestamp']].last().reset_index()

# Create user-item matrix (no normalization)
user_item_matrix = df.pivot(index='user_id', columns='parent_asin', values='rating').fillna(0)
sparse_matrix = csr_matrix(user_item_matrix.values)

# Save matrix for API use
joblib.dump(user_item_matrix, 'user_item_matrix.pkl')

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

    # Train SVD
    svd = TruncatedSVD(n_components=1000, random_state=42, n_iter=10)
    user_factors = svd.fit_transform(train_matrix)

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
            actual_ratings = actual[user_idx, top_k_items]
            precision = np.sum(actual_ratings >= 2) / k  # Threshold for 2/5
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
    mlflow.sklearn.log_model(svd, artifact_path="svd_model", signature=signature)
    mlflow.log_artifact('user_item_matrix.pkl')

    # Register model
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/svd_model"
    mlflow.register_model(model_uri, "BookRecommenderModel")

    print(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
    print(f"Train Precision@5: {train_precision:.4f}, Test Precision@5: {test_precision:.4f}")
