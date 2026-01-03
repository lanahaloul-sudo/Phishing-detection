# train_tabular_pipeline.py
import os
import json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score
)

from joblib import dump

RANDOM_STATE = 42

# عدّلي المسارات إذا لزم
TRAIN_PATH = "splits_features_train.csv"
VAL_PATH   = "splits_features_val.csv"
TEST_PATH  = "splits_features_test.csv"

MODEL_OUT  = "model_tabular_clean.joblib"
METRICS_OUT = "tabular_metrics.json"


def pick_label_col(df: pd.DataFrame) -> str:
    for c in ["label", "Label", "LABEL", "y", "target", "Target"]:
        if c in df.columns:
            return c
    raise KeyError(f"Label column not found. Columns are: {list(df.columns)[:30]}...")


def prepare_xy(df: pd.DataFrame, label_col: str):
    y = df[label_col].astype(int).values
    X = df.drop(columns=[label_col])

    # Keep only numeric columns (drop any accidental text columns)
    X = X.select_dtypes(include=[np.number]).copy()

    # If some numeric-looking columns are objects, force convert:
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = pd.to_numeric(X[col], errors="coerce")

    return X, y


def eval_split(name: str, pipe: Pipeline, X, y) -> dict:
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)[:, 1]

    rep = classification_report(y, pred, digits=4, output_dict=True)
    roc = roc_auc_score(y, proba)
    pr  = average_precision_score(y, proba)

    print(f"\n===== {name} RESULTS =====")
    print(classification_report(y, pred, digits=4))
    print(f"ROC-AUC = {roc:.6f}")
    print(f"PR-AUC  = {pr:.6f}")

    return {
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "report": rep
    }


def main():
    # 1) Load
    train_df = pd.read_csv("splits_tabular_train.csv")
    val_df   = pd.read_csv("splits_tabular_val.csv")
    test_df  = pd.read_csv("splits_tabular_test.csv")

    label_col = pick_label_col(train_df)

    # Ensure same label column exists in val/test
    if label_col not in val_df.columns or label_col not in test_df.columns:
        raise KeyError(f"Label col '{label_col}' not found in val/test. Check your split files.")

    # 2) Prepare X,y
    X_train, y_train = prepare_xy(train_df, label_col)
    X_val,   y_val   = prepare_xy(val_df, label_col)
    X_test,  y_test  = prepare_xy(test_df, label_col)

    # Align columns (important if any column missing)
    cols = list(X_train.columns)
    X_val  = X_val.reindex(columns=cols)
    X_test = X_test.reindex(columns=cols)

    print("Loaded splits:")
    print("Train:", X_train.shape, " Val:", X_val.shape, " Test:", X_test.shape)
    print("Label counts:")
    print("Train:", dict(pd.Series(y_train).value_counts().sort_index()))
    print("Val  :", dict(pd.Series(y_val).value_counts().sort_index()))
    print("Test :", dict(pd.Series(y_test).value_counts().sort_index()))

    # 3) Pipeline (Imputer + Scaler + ExtraTrees)
    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", ExtraTreesClassifier(
            n_estimators=600,
            max_depth=40,
            max_features="log2",
            min_samples_split=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    # 4) Train
    print("\nTraining TABULAR model...")
    pipe.fit(X_train, y_train)

    # 5) Evaluate
    metrics = {
        "val":  eval_split("VAL", pipe, X_val, y_val),
        "test": eval_split("TEST", pipe, X_test, y_test),
        "meta": {
            "train_path": TRAIN_PATH,
            "val_path": VAL_PATH,
            "test_path": TEST_PATH,
            "label_col": label_col,
            "n_features": int(X_train.shape[1]),
            "random_state": RANDOM_STATE
        }
    }

    # 6) Save model + metrics
    dump(pipe, MODEL_OUT)
    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Saved model -> {MODEL_OUT}")
    print(f"✅ Saved metrics -> {METRICS_OUT}")


if __name__ == "__main__":
    main()