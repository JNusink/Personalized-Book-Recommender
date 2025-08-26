import pandas as pd
df = pd.read_parquet('books_5core_1M.parquet')
print(f"Rows: {len(df)}, Users: {df['user_id'].nunique()}, Items: {df['parent_asin'].nunique()}")
