# Personalized Book Recommender

# Book Recommender System

A production-ready machine learning application that provides personalized book recommendations based on Amazon Books reviews. The system includes model training, versioning, REST API serving via FastAPI, a user-facing Streamlit frontend, MLflow experiment tracking, AWS cloud database integration, and deployment automation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Setup](#setup)
- [Running the Application](#running-the-application)
- [Docker Deployment](#docker-deployment)
- [Usage](#usage)
- [Architecture](#architecture)
- [Metrics](#metrics)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This project leverages a large-scale Amazon Books reviews dataset to build a personalized book recommendation system. The core backend model uses a TruncatedSVD-based collaborative filtering approach, served via FastAPI. The frontend allows users to query personalized recommendations. Experiment tracking and model versioning are managed with MLflow. Incoming requests and logs are stored in AWS DynamoDB for monitoring and caching.

---

## Setup

### Create and Activate Virtual Environment

**Windows (PowerShell):**

python -m venv venv
.\venv\Scripts\Activate.ps1

text

**Linux / macOS:**

python3 -m venv venv
source venv/bin/activate

text

### Install Dependencies

pip install -r requirements.txt

text

### Configure AWS Credentials

Obtain temporary AWS credentials from your instructor or AWS Educate portal. Set environment variables in PowerShell:

$env:AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY"
$env:AWS_SECRET_ACCESS_KEY="YOUR_SECRET_KEY"
$env:AWS_SESSION_TOKEN="YOUR_SESSION_TOKEN"
$env:AWS_DEFAULT_REGION="us-east-1"

text

Alternatively, use the AWS CLI:

aws configure

text

---

## Running the Application

### Start the FastAPI Server

Ensure virtual environment is active.

uvicorn app:app --host 0.0.0.0 --port 8000 --reload

text

Check health endpoint: [http://localhost:8000/health](http://localhost:8000/health)  
Expected response: `{"status": "healthy"}`

### Start the Streamlit Frontend

Open a new terminal with the active virtual environment:

streamlit run frontend.py

text

Access the frontend at [http://localhost:8501](http://localhost:8501)

### Start the MLflow Server (Optional)

For monitoring experiments and models:

mlflow server --host 0.0.0.0 --port 5000

text

Visit: [http://localhost:5000](http://localhost:5000)

---

## Docker Deployment (Optional)

Build and run the Docker container:

docker build -t book-recommender .
docker run -p 8000:8000 -p 8501:8501 book-recommender

text

Ensure your `Dockerfile` includes all dependencies and entrypoints for both backend and frontend.

---

## Usage

- Visit the Streamlit app at [http://localhost:8501](http://localhost:8501).
- Select a user ID from the dropdown (e.g., `AE224GVO7OHTYF26U6ER6BEVIUAQ`).
- Click **Get Recommendations** to view recommended books by ASIN.
- Alternatively, test the FastAPI endpoint directly using curl:

curl "http://localhost:8000/predict?user_id=AE224GVO7OHTYF26U6ER6BEVIUAQ"

text

Returns a JSON list of book recommendations.

---

## Architecture

- **Data Storage:** S3 bucket `my-book-recommender-2025-jtnusink/data/` stores model artifacts (`model.pkl`, `user_item_matrix.npz`, `all_recommendations.json`) and raw datasets.
- **Model:** TruncatedSVD model trained on a sparse user-item matrix, logged to MLflow model registry.
- **Backend:** FastAPI server on port 8000 serving `/predict` and `/health` routes; writes prediction logs to DynamoDB.
- **Frontend:** Streamlit app on port 8501 displaying precomputed recommendations.
- **Database:** DynamoDB table `BookRecommendations` with `user_id` (HASH) and `item_id` (RANGE) keys caching recommendations.
- **Monitoring:** MLflow UI for experiment and model performance tracking.

---

## Metrics

| Metric               | Training | Testing  |
|----------------------|----------|----------|
| RMSE                 | 0.0581   | 0.0654   |
| Precision @ 5        | 0.3541   | 0.2952   |
| MLflow Experiment    | `book_recommender` (version 14) |

---

## Troubleshooting

- **FastAPI Server Not Responding:**
  - Confirm server is running with `uvicorn app:app`.
  - Check no firewall blocks port 8000.
  - Verify `model.pkl` and `user_item_matrix.npz` exist in the project root.

- **Streamlit Frontend Fails:**
  - Ensure FastAPI backend is up.
  - Confirm `all_recommendations.json` exists and is accessible.
  - Restart with `streamlit run frontend.py --server.port 8501`.

- **AWS S3/DynamoDB Credential Issues:**
  - Renew AWS credentials if `ExpiredToken` errors occur.
  - Verify environment variables are correctly set (`echo %AWS_ACCESS_KEY_ID%` or `echo $AWS_ACCESS_KEY_ID`).
  - Match bucket and table names as configured.

- **Git Push Problems:**
  - Confirm branch is `main`.
  - Check remote URLs: `git remote -v`.
  - Use `git push -f` cautiously for branch sync issues.

---

## Contributing

We welcome contributions!

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-branch`.
3. Commit your changes: `git commit -m "Add feature"`.
4. Push to your fork: `git push origin feature-branch`.
5. Open a pull request.

Please ensure code quality and tests pass.

---

If you need help or have questions, please open an issue or contact the maintainers.
