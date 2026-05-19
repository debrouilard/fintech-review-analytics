import pandas as pd
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis", 
                              model="distilbert-base-uncased-finetuned-sst-2-english")

def get_sentiment(text):
    if not isinstance(text, str) or len(text.strip()) < 5:
        return "NEUTRAL", 0.0
    result = sentiment_pipeline(text[:512])[0]
    return result['label'], round(result['score'], 4)

def assign_themes(df):
    df = df.copy()
    lower = df['review'].str.lower()
    df['identified_theme'] = 'Other'
    
    df.loc[lower.str.contains('login|otp|password'), 'identified_theme'] = 'Account Access Issues'
    df.loc[lower.str.contains('slow|loading|transfer|crash|fail'), 'identified_theme'] = 'Transaction Performance'
    df.loc[lower.str.contains('ui|design|interface|look'), 'identified_theme'] = 'UI & Design'
    df.loc[lower.str.contains('support|help|customer'), 'identified_theme'] = 'Customer Support'
    df.loc[lower.str.contains('feature|add|please|request'), 'identified_theme'] = 'Feature Requests'
    return df