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
  
  required_cols = ["input_ids", "attention_mask"]
  for col in required_cols: 
    if col not in encodings:
      raise KeyError(f"Tokenizer output missing required field: {col}")

  return pd.DataFrame({
    "input_ids": encodings["input_ids"],
    "attention_mask": encodings["attention_mask"]
  })


# result = Tokenizer(data["review"], method="BERT")
# result = Tokenizer(data["review"], method="GPT-2")

# ========== BERT tokenizer mechanism ==========
# WORDS: One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. '
# 'They are right, as this is exactly what happened with me.The first thing that struck me about Oz was its '
# 'brutality and unflinching scenes of violence, which set in right from the word GO. Trust me, this is not a show for the '
# 'faint hearted or timid. This show pulls no punches with regards to drugs, sex or violence. Its is hardcore, in the classic '
# 'use of the word.It is called OZ as that is the nickname given to the Oswald Maximum Security State Penitentary. It focuses mainly '
# 'on Emerald City, an experimental section of the prison where all the cells have glass fronts and face inwards, so privacy is not '
# 'high on the agenda. Em City is home to many..Aryans, Muslims, gangstas, Latinos, Christians, Italians, Irish and more....so scuffles, '
# 'death stares, dodgy dealings and shady agreements are never far away.I would say the main appeal of the show is due to the fact that '
# 'it goes where other shows wouldn't dare. Forget pretty pictures painted for mainstream audiences, forget charm, forget romance...OZ doesn't mess around. The first episode '
# 'I ever saw struck me as so nasty it was surreal, I couldn't say I was ready for it, but as I watched more, I developed a taste for Oz, and got accustomed to the high levels of graphic violence. 
# Not just violence, but injustice (crooked guards who'll be sold out for a nickel, inmates who'll kill on order and get away with it, well mannered, middle class inmates being turned into prison bitches 
#                                   due to their lack of street skills or prison experience) Watching Oz, you may become comfortable with what is uncomfortable viewing....thats if you can get in touch with your darker side. 
# -> TOKENS: ['one', 'of', 'the', 'other', 'reviewers', 'has', 'mentioned', 'that', 'after', 'watching', 'just', '1', 'oz', 'episode', 'you', "'", 'll', 'be', 'hooked', '.', 'they', 'are', 'right', ',', 'as', 'this', 'is', 'exactly', 
#             'what', 'happened', 'with', 'me', '.', 'the', 'first', 'thing', 'that', 'struck', 'me', 'about', 'oz', 'was', 'its', 'brutality', 'and', 'un', '##fl', '##in', '##ching', 'scenes']



# ========== GPT-2 tokenizer mechanism ==========
# WORDS: One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right, as this is exactly what happened with me.The first thing that struck me about Oz '
# 'was its brutality and unflinching scenes of violence, which set in right from the word GO. Trust me, this is not a show for the faint hearted or timid. This show pulls no punches with regards to drugs, '
# 'sex or violence. Its is hardcore, in the classic use of the word.It is called OZ as that is the nickname given to the Oswald Maximum Security State Penitentary. It focuses mainly on Emerald City, '
# 'an experimental section of the prison where all the cells have glass fronts and face inwards, so privacy is not high on the agenda. Em City is home to many..Aryans, Muslims, gangstas, Latinos, Christians, '
# 'Italians, Irish and more....so scuffles, death stares, dodgy dealings and shady agreements are never far away.I would say the main appeal of the show is due to the fact that it goes where other shows wouldn't dare. 
# Forget pretty pictures painted for mainstream audiences, forget charm, forget romance...OZ doesn't mess around. The first episode I ever saw struck me as so nasty it was surreal, I couldn't say I was ready for it, but as 
# I watched more, I developed a taste for Oz, and got accustomed to the high levels of graphic violence. Not just violence, but injustice (crooked guards who'll be sold out for a nickel, inmates who'll kill on order and 
# get away with it, well mannered, middle class inmates being turned into prison bitches due to their lack of street skills or prison experience) Watching Oz, you may become comfortable with 
# what is uncomfortable viewing....thats if you can get in touch with your darker side. 
# -> TOKENS: ['One', 'Ġof', 'Ġthe', 'Ġother', 'Ġreviewers', 'Ġhas', 'Ġmentioned', 'Ġthat', 'Ġafter', 'Ġwatching', 'Ġjust', 'Ġ1', 'ĠOz', 'Ġepisode', 'Ġyou', "'ll", 'Ġbe', 'Ġhooked', '.', 
# 'ĠThey', 'Ġare', 'Ġright', ',', 'Ġas', 'Ġthis', 'Ġis', 'Ġexactly', 'Ġwhat', 'Ġhappened', 'Ġwith', 'Ġme', '.', 'The', 'Ġfirst', 'Ġthing', 'Ġthat', 'Ġstruck', 'Ġme', 'Ġabout', 'ĠOz', 
# 'Ġwas', 'Ġits', 'Ġbrutality', 'Ġand', 'Ġunfl', 'inch', 'ing', 'Ġscenes', 'Ġof', 'Ġviolence']