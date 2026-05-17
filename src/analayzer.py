"""
Partial Task 2: Sentiment Analysis + Keyword Extraction
"""

from transformers import pipeline
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def get_sentiment(text: str):
    if pd.isna(text) or len(str(text).strip()) < 3:
        return "NEUTRAL", 0.0
    result = sentiment_pipeline(str(text)[:512])[0]
    label = result['label']  # POSITIVE / NEGATIVE
    score = result['score']
    return label, round(score, 4)

def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    print("Running DistilBERT sentiment analysis...")
    results = df['review'].apply(get_sentiment)
    df['sentiment_label'] = results.apply(lambda x: x[0])
    df['sentiment_score'] = results.apply(lambda x: x[1])
    return df

def extract_keywords(df: pd.DataFrame, top_n: int = 15):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=100)
    tfidf = vectorizer.fit_transform(df['review'].dropna())
    words = vectorizer.get_feature_names_out()
    sums = tfidf.sum(axis=0)
    top = [(words[i], sums[0,i]) for i in sums.argsort()[0, -top_n:][::-1]]
    return top

# Example usage in notebook
if __name__ == "__main__":
    df = pd.read_csv("data/cleaned/reviews_cleaned.csv")
    df = add_sentiment(df)
    print(df['sentiment_label'].value_counts())
    print("\nTop Keywords:", extract_keywords(df))