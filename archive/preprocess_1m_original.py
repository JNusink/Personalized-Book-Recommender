import pandas as pd

# Process 18GB Books.jsonl to create 1M 5-core dataset
chunk_size = 1000000
chunks = []
for chunk in pd.read_json('Books.jsonl', lines=True, chunksize=chunk_size, nrows=2000000):
    chunks.append(chunk[['user_id', 'parent_asin', 'rating', 'timestamp']])  # Keep timestamp
df = pd.concat(chunks).sample(n=1000000, random_state=42)
df = df.sort_values('timestamp').groupby(['user_id', 'parent_asin'])[['rating', 'timestamp']].last().reset_index()
df = df.groupby('user_id').filter(lambda x: len(x) >= 5)
df = df.groupby('parent_asin').filter(lambda x: len(x) >= 5)
df.to_parquet('books_5core_1M.parquet')
print("Created books_5core_1M.parquet")
