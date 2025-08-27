# Personalized Book Recommender

## Overview

This project is a complete MLOps system for a personalized book recommendation engine using **Singular Value Decomposition (SVD)** for collaborative filtering. It processes book rating data, trains an SVD model, deploys a FastAPI backend for serving recommendations, and a Streamlit frontend for user interaction. Both components are containerized with Docker for scalable deployment on AWS EC2. The system is optimized for limited resources (e.g., t2.medium instance with 4GB RAM) by sampling data and reducing model complexity, leveraging data from local Parquet or S3 JSONL with fallback.

This README provides a detailed guide for setup, deployment, example requests, and troubleshooting, enabling replication by cloning the repository and downloading data from S3 or recreating it from the original Amazon Reviews 2023 dataset.

---

## Key Features

- Data loading from local Parquet or S3 JSONL streaming fallback.
- Model training with MLflow logging and artifact storage.
- FastAPI backend and Streamlit frontend, both containerized with Docker.
- Professor-ready: all SSH terminal steps, troubleshooting, and security placeholders (`your_access_key_here`).

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Data Acquisition](#data-acquisition)
- [Local Setup](#local-setup)
- [EC2 Setup](#ec2-setup)
- [Example Requests](#example-requests)
- [Project Performance Metrics](#project-performance-metrics)
- [Project Phases](#project-phases)
- [Troubleshooting](#troubleshooting)
- [TODO](#todo)
- [License](#license)

---

## Prerequisites

- **Local Machine**: Windows (PowerShell), VS Code, Python 3.10+ (NumPy 2.x and scikit-learn 1.7.0).
- **AWS Account**: AWS Academy Sandbox or personal account with EC2, S3, IAM permissions.
- **GitHub Repository**:  
  Clone:  
git clone https://github.com/JNusink/Personalized-Book-Recommender.git

text
- **Dependencies**: Listed in `requirements.txt`.
- **Data**:  
Download `books_5core_1M.parquet` or `Books.jsonl` from S3 (`s3://my-book-recommender-2025-jtnusink/data/`), or recreate from Amazon Reviews 2023 (see [Data Acquisition](#data-acquisition)).
- **SSH Key**: Generate a `.pem` file (e.g., `book-recommender-key.pem`) for EC2 access.
- **Docker**: Installed on EC2 and locally for containerization.

---

## Data Acquisition

### Download from S3 (Preferred Method)

Configure your AWS credentials, then run:
aws s3 cp s3://my-book-recommender-2025-jtnusink/data/books_5core_1M.parquet .
aws s3 cp s3://my-book-recommender-2025-jtnusink/data/Books.jsonl .

text

**Troubleshooting:**
- Access denied? Verify IAM policy includes S3 read access.
- For connection test:  
aws s3 ls s3://my-book-recommender-2025-jtnusink/data/

text

### Alternative: Recreate from Amazon Reviews 2023

**1. Download the Dataset:**
- Visit [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) and download the Books category dataset (Books.jsonl or equivalent).

**Troubleshooting:**  
If the link is broken, search "Amazon Reviews 2023 McAuley Lab" or check for mirrors. The file is ~18GB; ensure a stable connection and at least 20GB disk space.

**2. Process the Dataset:**
- Place `Books.jsonl` in your project directory.
- Create `process_data.py`:
import pandas as pd
import jsonlines

Load JSONL
data = []
with jsonlines.open('Books.jsonl') as reader:
for obj in reader:
data.append(obj)
df = pd.DataFrame(data)

Filter for 5-core (users/items with at least 5 ratings)
user_counts = df['user_id'].value_counts()
item_counts = df['parent_asin'].value_counts()
df = df[df['user_id'].isin(user_counts[user_counts >= 5].index)]
df = df[df['parent_asin'].isin(item_counts[item_counts >= 5].index)]

Sample 1M rows if dataset exceeds, else use all
df = df.sample(n=1000000, random_state=42) if len(df) > 1000000 else df

Save as Parquet
df.to_parquet('books_5core_1M.parquet', index=False)

text

Run:
python process_data.py

text

**Troubleshooting:**
- Install jsonlines if needed: `pip install jsonlines`
- Reduce n=1000000 if memory issues occur.
- Ensure sufficient disk space.

---

## Local Setup

**1. Clone the Repository:**
git clone https://github.com/JNusink/Personalized-Book-Recommender.git
cd Personalized-Book-Recommender

text
> *If Git is not installed, download from [git-scm.com](https://git-scm.com/).*

**2. Set Up Virtual Environment:**
python -m venv venv
& "venv\Scripts\Activate.ps1"

text

**3. Install Dependencies:**
pip install -r requirements.txt

text

**Troubleshooting:**
- Dependency conflicts: pin packages (`numpy==1.24.3`, etc.).
- Windows build errors: install Visual C++ Build Tools.
- Missing jsonlines: `pip install jsonlines`.

**4. Run Locally:**

- **Train the Model:**
python train_model.py

text
- **Run FastAPI Backend:**
uvicorn app:app --host 0.0.0.0 --port 8000

text
- **Run Streamlit Frontend:**
streamlit run frontend.py

text

**Troubleshooting:**
- If `model.pkl` is missing, rerun `train_model.py`.
- For memory errors, add sampling in `train_model.py` (e.g., `df = df.sample(n=20000, random_state=42)`).

---

## EC2 Setup

**1. Launch EC2 Instance**
- AMI: Amazon Linux 2023.
- Type: t2.medium (4GB RAM).
- Key Pair: Generate (e.g., `book-recommender-key.pem`).
- Security Group: open ports 22 (SSH), 8000, and 8501.
- Assign public IP.

**Troubleshooting:**
- Launch fails? Check quota/subnet, public subnet required.

**2. Connect via SSH:**
ssh -i book-recommender-key.pem ec2-user@your-ec2-ip

text

**3. Install System Dependencies:**
sudo yum update -y
sudo yum install git docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user

Log out and reconnect for group permission changes
text

**4. Clone Repo and Install Python Dependencies:**
git clone https://github.com/JNusink/Personalized-Book-Recommender.git
cd Personalized-Book-Recommender
pip3 install -r requirements.txt --user

text

**5. Configure AWS Credentials:**
export AWS_ACCESS_KEY_ID="your_access_key_here"
export AWS_SECRET_ACCESS_KEY="your_secret_key_here"
export AWS_SESSION_TOKEN="your_session_token_here"
export AWS_DEFAULT_REGION="us-east-1"

text
> Test with:  
> `aws s3 ls s3://my-book-recommender-2025-jtnusink/data/`

**6. Download Data from S3:**
aws s3 cp s3://my-book-recommender-2025-jtnusink/data/books_5core_1M.parquet .
aws s3 cp s3://my-book-recommender-2025-jtnusink/data/Books.jsonl .

text

**7. Train the Model:**
python3 train_model.py

text
- Memory error? Sample smaller data in script, monitor with `top`.

**8. Containerization:**
docker build -t book-recommender .
docker build -f Dockerfile.frontend -t book-recommender-frontend .
docker run -d -p 8000:8000 book-recommender
docker run -d -p 8501:8501 book-recommender-frontend

text
> To stop: `docker ps` then `docker stop <id>`

**9. Test Deployment:**
- Backend local: `curl http://localhost:8000/health`
- Backend external: `curl http://your-ec2-ip:8000/health`
- Frontend: `http://your-ec2-ip:8501`

---

## Example Requests

### Backend Health Check

- **Endpoint**: `/health`
- **Method**: GET
- **Usage**:
curl http://your-ec2-ip:8000/health

text
- **Response**: `{"status":"healthy"}`
- **Troubleshooting**: If 500 error, check `model.pkl` via `docker logs`.

---

### Backend Recommendation

- **Endpoint**: `/recommend`
- **Method**: GET
- **Parameters**: `user_id`
- **Usage**:
curl http://your-ec2-ip:8000/recommend?user_id=example_user_id

text
- **Response**: `{"recommendations": ["book1", "book2"]}`
- **Troubleshooting**: Check `app.py` logic or refresh `all_recommendations.json`.

---

### Frontend Access

- **URL**: `http://your-ec2-ip:8501`
- **View**: Streamlit UI in browser

---

## Project Performance Metrics

| Model Version | Train RMSE | Test RMSE | Train Precision@5 | Test Precision@5 | Notes                        |
|---------------|------------|-----------|-------------------|------------------|------------------------------|
| 18 (Optimized)| 0.0502     | 0.0524    | 0.0734            | 0.0497           | 20K rows, 200 SVD components |

*Optimized for t2.medium with 20,000 rows and `n_components=200`.*

---

## Project Phases

- **Phase 1–4**: Completed (data processing, modeling, backend, CI/CD).
- **Phase 5**: Completed (containerization, deployment).

---

## Troubleshooting

- **Memory Error in Training**: Add data sampling in `train_model.py` (e.g. `df = df.sample(n=10000)`), monitor EC2 memory.
- **Dependency Conflicts**: Pin to compatible versions (e.g., `numpy==1.24.3`).
- **S3 Download Failure**: Test S3 permissions, use local data if blocked.
- **Docker Build Failure**: Verify Dockerfile, clean `requirements.txt`.
- **Port Conflict**: Free with `docker ps`/`docker stop <id>`.
- **External Timeout**: Security group must allow ports 8000, 8501.
- **Model Loading Error**: Rerun training, verify Docker build includes model.

---

## TODO

- Enhance frontend UI.
- Deploy backend and frontend on separate EC2 instances.
- Improve precision with larger sample size on a more capable instance.

---

If you have issues, open a GitHub issue or contact the maintainer.
