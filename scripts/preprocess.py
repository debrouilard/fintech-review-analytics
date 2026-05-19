import pandas as pd

print("Starting preprocessing...")
df = pd.read_csv("data/raw/reviews_raw.csv")
print(f"Raw rows: {len(df)}")

# Task 1 requirements
df = df.drop_duplicates(subset=['review_id'])
df = df.dropna(subset=['review', 'rating'])
df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

df = df[['review', 'rating', 'date', 'bank', 'source']]
df.to_csv("data/cleaned/reviews_cleaned.csv", index=False)
print(f"✅ Preprocessed {len(df)} reviews")