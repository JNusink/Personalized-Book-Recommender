# Personalized Book Recommender

## Overview
A personalized book recommendation system using SVD, developed as part of an MLOps project.

## Setup Instructions
- Clone the repo at https://github.com/JNusink/Personalized-Book-Recommender.git.
- Set up a virtual environment: `python -m venv venv`.
- Activate it: `venv\Scripts\Activate.ps1` (Windows).
- Install dependencies: `pip install -r requirements.txt`.
- Use `Dockerfile` and `.github/workflows/ci.yml` for deployment and CI.

## Usage
- Run the FastAPI server: `uvicorn app:app --host 0.0.0.0 --port 8000`.
- Launch the Streamlit frontend: `streamlit run frontend.py`.
- Start the monitoring dashboard: `streamlit run monitoring.py --server.port 8502`.

## Performance Metrics (Model Version 18)
- Train RMSE: 0.0581
- Test RMSE: 0.0654
- Train Precision@5: 0.3541
- Test Precision@5: 0.2952

## Project Phases
- **Phase 1-3**: Completed (data prep, model training, monitoring setup).
- **Phase 4**: CI/CD implemented and verified (successfully passed on 2025-08-26).
- **Phase 5**: Deployment and optimization (in progress).

## TODO
- Add detailed deployment instructions.
- Include EC2 optimization steps.
