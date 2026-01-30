import pandas as pd
import os 

def load_data(): 
  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  DATA_PATH = os.path.join(BASE_DIR, "IMDB_Dataset.csv")

  df = pd.read_csv(DATA_PATH, encoding="utf-8")
  # Normalize sentiment value into integer
  df['sentiment'] = df["sentiment"].replace(to_replace=['positive', 'negative'], value=[0, 1])

  # Clean data which is missing value
  df = df.dropna(subset=['review']).reset_index(drop=True)
  # Normalize the "<br /> syntax in html"
  df["review"] = df["review"].str.replace(r'<br\s*/?>', '', regex=True)

  #Calculate average words per sentence
  df['word_count'] = df["review"].apply(lambda x: len(x.split()))
  avg_words = df["word_count"].mean() 
  df = df.drop(columns=["word_count"])

  print("\n\n" + 10*"=" + " Load IMDB dataset " + 10*"=")
  print(f"Data overvew: \n{df.head()}")
  print(f"Average word per sentence: {avg_words:.2f}")

  return df
