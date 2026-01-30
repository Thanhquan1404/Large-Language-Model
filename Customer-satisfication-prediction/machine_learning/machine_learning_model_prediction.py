from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time

def data_processing(
    max_numsrow=2000, 
    embedding_data_path="../data/embedded_data/BERT_embedded_result.csv",
    label_data_path="../data/IMDB_Dataset.csv"
):
    # Load embedding (drop index column if exists)
    X = pd.read_csv(
        embedding_data_path,
        encoding="utf-8",
        nrows=max_numsrow,
        index_col=0
    )

    # Load label
    y = pd.read_csv(
        label_data_path,
        encoding="utf-8",
        nrows=max_numsrow
    )["sentiment"].replace({
       'positive': 0,
       'negative': 1,
    })

    # Reset index for safety
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    if len(X) != len(y):
        raise ValueError(
            f"Length mismatch: X={len(X)}, y={len(y)}"
        )

    return X, y


def run_model_pipeline(X, y, output_file="./model_result.csv"):
  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

  models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest-Neighbors": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "CatBoost": CatBoostClassifier(verbose=0),
  }

  results = []

  print("\n\n" + 10*"=" + "Running machine learning model section" + 10*"=" + "\n")

  for name, model in models.items():
    print(10*"=" + f"Running model name ({name})" + 10*"=" + "\n")
    start_time = time.time()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    duration = time.time() - start_time
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    recall = recall_score(y_test, preds)
    precision = precision_score(y_test, preds)
    
    print(f"{name:20} | Acc: {acc:.4f} | F1: {f1:.4f} | Time: {duration:.2f}s\n")
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "F1_Score": f1,
        "Recall": recall,
        "Precision": precision,
        "Training_Time_Sec": duration,
    })

  results_df = pd.DataFrame(results)
  results_df.to_csv(output_file, index=False)
  print(f"Successfull to save results into {output_file} \n\n")


# tokenize_mechanism = "BERT"
# embedding_data_path = f"../data/embedded_data/{tokenize_mechanism}_embedded_result.csv"
# X, y = data_processing(embedding_data_path=embedding_data_path)
# run_model_pipeline(X, y, output_file=f"./{tokenize_mechanism}_result.csv")

tokenize_mechanism = "GPT-2"
embedding_data_path = f"../data/embedded_data/{tokenize_mechanism}_embedded_result.csv"
X, y = data_processing(embedding_data_path=embedding_data_path)
run_model_pipeline(X, y, output_file=f"./{tokenize_mechanism}_result.csv")