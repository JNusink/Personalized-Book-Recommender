# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose ports for FastAPI, Streamlit, and MLflow
EXPOSE 8000 8501 5000

# Run the application (FastAPI, Streamlit, and MLflow concurrently)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 & streamlit run frontend.py --server.port 8501 --server.address 0.0.0.0 & mlflow server --host 0.0.0.0 --port 5000"]