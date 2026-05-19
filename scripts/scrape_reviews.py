import time
import pandas as pd
from google_play_scraper import reviews, Sort
from tqdm import tqdm

def scrape_bank(app_id: str, bank_name: str, target: int = 450):
    print(f"Scraping {bank_name}...")
    all_reviews = []
    token = None
    try:
        while len(all_reviews) < target:
            result, token = reviews(app_id, lang='en', country='et', 
                                  sort=Sort.NEWEST, count=200, continuation_token=token)
            all_reviews.extend(result)
            if not token: 
                break
            time.sleep(1.5)
        df = pd.DataFrame(all_reviews)
        df = df[['reviewId','content','score','at']].copy()
        df.rename(columns={'content':'review','score':'rating','at':'date'}, inplace=True)
        df['bank'] = bank_name
        df['source'] = 'Google Play'
        df['review_id'] = df['reviewId']
        return df
    except Exception as e:
        print(f"Error scraping {bank_name}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    banks = [
        ("com.combanketh.mobilebanking", "Commercial Bank of Ethiopia"),
        ("com.boa.boaMobileBanking", "Bank of Abyssinia"),
        ("com.dashen.dashensuperapp", "Dashen Bank")
    ]
    dfs = []
    for app_id, name in banks:
        df = scrape_bank(app_id, name)
        if not df.empty:
            dfs.append(df)
        time.sleep(2)
    
    pd.concat(dfs, ignore_index=True).to_csv("data/raw/reviews_raw.csv", index=False)
    print("✅ Scraping completed!")