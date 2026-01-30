import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import BertTokenizer, GPT2TokenizerFast
from data.read_data import load_data
import pandas as pd

data = load_data()

def print_token(tokenizer, text, max_tokens=50):
  tokens = tokenizer.tokenize(text)
  print(f"WORDS: {text} \n-> TOKENS: {tokens[:max_tokens]}")

def Tokenizer(texts, method="BERT", max_length=128):
  if (isinstance(texts, pd.Series)):
    texts = texts.to_list()

  print("\n\n" + 10*"=" + f" {method} tokenizer mechanism " + 10*"=")

  if (method == "BERT"):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    print_token(tokenizer, texts[0])
  elif (method == "GPT-2"): 
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token #IMPORTANT
    print_token(tokenizer, texts[0])
  else:
    raise ValueError("Unidentified tokenizer mechanism!")
  
  encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors=None  # keep as Python lists
    )

  tokens_df = pd.DataFrame(encodings)

  return tokens_df


Tokenizer(data["review"], method="BERT")
Tokenizer(data["review"], method="GPT-2")