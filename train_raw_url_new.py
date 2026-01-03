import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from joblib import dump

RANDOM_STATE = 42

TRAIN_PATH = "splits/train.csv"
VAL_PATH   = "splits/val.csv"
TEST_PATH  = "splits/test.csv"

MODEL_OUT  = "model_raw_new.joblib"

def pick_url_col(df):
    # يلقط url_raw إذا موجود، وإلا url
    for c in ["url_raw", "url", "URL", "Url_raw", "URL_raw"]:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find URL column. Columns: {list(df.columns)[:30]}")

def load_split(path):
    df = pd.read_csv(path)
    url_col = pick_url_col(df)
    if "label" not in df.columns:
        raise KeyError(f"'label' column not found in {path}. Columns: {list(df.columns)[:30]}")
    X = df[url_col].astype(str).fillna("")
    y = df["label"].astype(int)
    return X, y, url_col

def evaluate(name, model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    print(f"\n==== {name} ====")
    print(classification_report(y, pred, digits=4))

    roc = roc_auc_score(y, proba)
    pr  = average_precision_score(y, proba)  # PR-AUC
    print(f"ROC-AUC = {roc:.6f}")
    print(f"PR-AUC  = {pr:.6f}")
    return roc, pr

def main():
    X_train, y_train, col_train = load_split(TRAIN_PATH)
    X_val,   y_val,   col_val   = load_split(VAL_PATH)
    X_test,  y_test,  col_test  = load_split(TEST_PATH)

    print(f"Loaded splits.")
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"URL col (train/val/test): {col_train} / {col_val} / {col_test}")

    # RAW pipeline: char-level TF-IDF + Logistic Regression (balanced)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 6),
            min_df=2,
            max_features=250000,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=None
        ))
    ])

    print("\nTraining RAW URL model...")
    pipe.fit(X_train, y_train)

    # Validate & Test
    evaluate("VAL", pipe, X_val, y_val)
    evaluate("TEST", pipe, X_test, y_test)

    dump(pipe, MODEL_OUT)
    print(f"\nSaved model -> {MODEL_OUT}")

if __name__ == "__main__":
    main()