import pandas as pd
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Starting data loading and preprocessing of Books.jsonl")

# Check if input file exists
input_file = 'Books.jsonl'
if not os.path.exists(input_file):
    logger.error(f"{input_file} not found in {os.getcwd()}")
    raise FileNotFoundError(f"{input_file} not found")

# Initialize chunk list
chunks = []

# Process Books.jsonl in chunks with error handling
try:
    for chunk in pd.read_json(input_file, lines=True, chunksize=500000):
        logger.debug(f"Processing chunk of size {len(chunk)}")
        if 'reviewerID' not in chunk or 'asin' not in chunk or 'overall' not in chunk or 'unixReviewTime' not in chunk:
            logger.warning("Skipping chunk due to missing required columns")
            continue
        chunk = chunk.rename(columns={'reviewerID': 'user_id', 'asin': 'parent_asin', 'overall': 'rating', 'unixReviewTime': 'timestamp'})
        chunks.append(chunk[['user_id', 'parent_asin', 'rating', 'timestamp']])
    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"Loaded {len(df)} rows from {input_file}")
except ValueError as e:
    logger.error(f"Invalid JSON in {input_file}: {e}")
    raise
except Exception as e:
    logger.error(f"Error reading {input_file}: {e}")
    raise

# Sample 1M rows for a robust dataset
if len(df) < 1000000:
    logger.warning("Dataset has fewer than 1M rows, using all available data")
df = df.sample(n=min(1000000, len(df)), random_state=42)
logger.info(f"Sampled {len(df)} rows")

# Ensure ratings are un-normalized (1-5) and filter valid ratings
if df['rating'].max() <= 1.0:
    df['rating'] = df['rating'] * 5
df = df[df['rating'] > 0]
logger.info(f"Rating range after normalization: {df['rating'].min()} - {df['rating'].max()}")

# Convert timestamp if needed (assuming unix time to seconds)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

# Aggregate by most recent rating
df = df.sort_values('timestamp').groupby(['user_id', 'parent_asin'])[['rating', 'timestamp']].last().reset_index()
logger.info(f"Aggregated to {len(df)} unique user-item pairs")

# Apply 5-core filtering
user_counts = df['user_id'].value_counts()
item_counts = df['parent_asin'].value_counts()
df = df[df['user_id'].isin(user_counts[user_counts >= 5].index)]
df = df[df['parent_asin'].isin(item_counts[item_counts >= 5].index)]
logger.info(f"Applied 5-core filtering, resulting in {len(df)} rows")

# Save as Parquet for efficiency
output_path = 'books_5core_1M.parquet'
df.to_parquet(output_path, index=False)
logger.info(f"Saved to {output_path} with size {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")

# Validate output
if os.path.exists(output_path):
    loaded_df = pd.read_parquet(output_path)
    if len(loaded_df) == len(df):
        logger.info("Validation successful: Output file matches input DataFrame")
    else:
        logger.warning("Validation failed: Output file size mismatch")
else:
    logger.error("Output file not created")
