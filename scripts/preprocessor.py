# scripts/preprocess.py
import pandas as pd

df = pd.read_csv("data/raw/reviews_raw.csv")
df = df.drop_duplicates(subset=['review_id'])
df = df.dropna(subset=['review', 'rating'])
df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

df[['review','rating','date','bank','source']].to_csv("data/cleaned/reviews_cleaned.csv", index=False)
print(f"Preprocessed {len(df)} reviews")