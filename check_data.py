import pandas as pd

# Load the 5-core dataset
df = pd.read_parquet('books_5core_500k.parquet')
print(df.info())  # Column names and types
print(df['rating'].value_counts())  # Rating distribution
print(f"Unique users: {df['user_id'].nunique()}, Unique items: {df['item_id'].nunique()}")

import pandas as pd

# Load data
df = pd.read_parquet('books_5core_500k.parquet')

# Rating distribution
print("Rating Distribution:")
print(df['rating'].value_counts(normalize=True).sort_index())

# Unique users and items
n_users = df['user_id'].nunique()
n_items = df['item_id'].nunique()
print(f"Unique users: {n_users}")
print(f"Unique items: {n_items}")

# Sparsity
n_ratings = len(df)
sparsity = 1 - (n_ratings / (n_users * n_items))
print(f"Sparsity: {sparsity:.4f} ({n_ratings} ratings / {n_users * n_items} possible)")