import boto3
import os

if not os.path.exists('Books.jsonl'):
    print("Error: Books.jsonl not found in C:\\VS code\\Final project")
    exit(1)

s3 = boto3.client('s3')
bucket_name = 'my-book-recommender-2025-jtnusink'
s3.upload_file('Books.jsonl', bucket_name, 'data/Books.jsonl')
print(f"Uploaded Books.jsonl to s3://{bucket_name}/data/Books.jsonl")