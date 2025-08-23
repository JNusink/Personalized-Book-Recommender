import pandas as pd
import json

# Read Books.jsonl incrementally and sample
chunk_size = 500000  # Larger chunk size due to 64GB RAM
chunks = []
for chunk in pd.read_json('Books.jsonl', lines=True, chunksize=chunk_size, nrows=2000000):
    chunks.append(chunk)
df = pd.concat(chunks)
df = df.sample(n=500000, random_state=42)  # Sample 500k reviews for robust baseline
print(df.head())
print(f"Dataset size: {df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")

# Rename columns for consistency
df = df.rename(columns={'reviewerID': 'user_id', 'asin': 'item_id', 'overall': 'rating'})

# Filter to 5-core (users and items with ≥5 reviews)
df = df.groupby('user_id').filter(lambda x: len(x) >= 5)
df = df.groupby('item_id').filter(lambda x: len(x) >= 5)
print(f"5-core dataset size: {len(df)} rows")

# Save as Parquet for efficiency
df.to_parquet('books_5core_500k.parquet')
