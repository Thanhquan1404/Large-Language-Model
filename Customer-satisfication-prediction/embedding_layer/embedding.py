import os
import sys
import torch
import pandas as pd
from transformers import BertModel, GPT2Model

# Ensure project root is available
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def get_embedding_features(tokens_df, method="BERT", device=None):
    """
    Transform tokenized inputs into embedding features.

    Parameters
    ----------
    tokens_df : pd.DataFrame
        Must contain 'input_ids' and 'attention_mask'
    method : str
        'BERT' or 'GPT-2'
    device : torch.device or None

    Returns
    -------
    pd.DataFrame
        Embedding features
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_ids = torch.tensor(tokens_df["input_ids"].tolist()).to(device)
    attention_mask = torch.tensor(tokens_df["attention_mask"].tolist()).to(device)

    print("\n\n" + 10*"=" + f"Running embedding layer with {method}" + 10*"=")
    if method == "BERT":
        model = BertModel.from_pretrained("bert-base-uncased").to(device)
        model.eval()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            features = outputs.pooler_output  # (batch_size, 768)

    elif method == "GPT-2":
        model = GPT2Model.from_pretrained("gpt2").to(device)
        model.eval()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_states = outputs.last_hidden_state
            features = torch.mean(last_hidden_states, dim=1)

    else:
        raise ValueError("method must be either 'BERT' or 'GPT-2'")

    # Convert to DataFrame for ML + CSV compatibility
    features_df = pd.DataFrame(features.cpu().numpy())
    return features_df

from tokenize_layer.tokenizer import Tokenizer
from data.read_data import load_data

data = load_data()

tokens = Tokenizer(data["review"], method="BERT")

bert_embeddings = get_embedding_features(tokens, method="BERT")

BERT_output_path = "../data/embedded_data/BERT_embedded_result.csv"
bert_embeddings.to_csv(BERT_output_path, index=False)

