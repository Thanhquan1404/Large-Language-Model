import pandas as pd
import os 

def load_data(max_size=2000, max_words=150):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "IMDB_Dataset.csv")

    df = pd.read_csv(DATA_PATH, encoding="utf-8", nrows=max_size)

    df['sentiment'] = df['sentiment'].replace({
        'positive': 0,
        'negative': 1
    })

    df = df.dropna(subset=['review']).reset_index(drop=True)

    df["review"] = (
        df["review"]
        .str.replace(r'<br\s*/?>', '', regex=True)
        .apply(lambda x: " ".join(x.split()[:max_words]))
    )

    avg_words = df["review"].apply(lambda x: len(x.split())).mean()

    print("\n========== Load IMDB dataset ==========")
    print(df.head())
    print(f"Average word per sentence: {avg_words:.2f}")

    return df

