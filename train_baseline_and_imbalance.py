import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier

# SMOTE (اختياري - الطريقة 2)
try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_OK = True
except ImportError:
    IMBLEARN_OK = False


# -----------------------------
# Helpers
# -----------------------------
def print_dist(name, y):
    y = pd.Series(y)
    counts = y.value_counts().to_dict()
    total = len(y)
    phish = counts.get(1, 0)
    ratio = phish / total if total else 0
    print(f"\n{name}: n={total} | label_counts={counts} | phish_ratio={ratio:.4f}")

def evaluate(model, X, y, name="VAL"):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    print(f"\n===== {name} RESULTS =====")
    print(classification_report(y, pred, digits=4))

    # AUCs (مفيدة جداً للـ imbalance)
    roc = roc_auc_score(y, proba)
    pr = average_precision_score(y, proba)
    print(f"ROC-AUC = {roc:.6f}")
    print(f"PR-AUC  = {pr:.6f}")
    return {"roc_auc": roc, "pr_auc": pr}

# -----------------------------
# Main
# -----------------------------
def main():
    DATA = "features_dataset.csv"

    # نسب التقسيم
    TEST_SIZE = 0.20
    VAL_SIZE_FROM_REMAIN = 0.20  # يعني: 20% من الباقي بعد test => ~16% من الكل
    RANDOM_STATE = 42

    df = pd.read_csv(DATA)

    # label لازم تكون موجودة
    if "label" not in df.columns:
        raise KeyError("Column 'label' not found in features_dataset.csv")

    # X = كل الأعمدة عدا label
    X = df.drop(columns=["label"])
    y = df["label"].astype(int)

    # تأكد إن كلشي رقمي
    non_numeric = [c for c in X.columns if not np.issubdtype(X[c].dtype, np.number)]
    if non_numeric:
        raise ValueError(f"Non-numeric columns found in X: {non_numeric}\n"
                         f"Fix feature extraction so all features are numeric.")

    # -----------------------------
    # 1) Split: Train / Val / Test
    # -----------------------------
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=VAL_SIZE_FROM_REMAIN,
        stratify=y_trainval, random_state=RANDOM_STATE
    )

    print_dist("TRAIN", y_train)
    print_dist("VAL", y_val)
    print_dist("TEST", y_test)

    # حفظ splits (اختياري)
    train_out = X_train.copy()
    train_out["label"] = y_train.values
    val_out = X_val.copy()
    val_out["label"] = y_val.values
    test_out = X_test.copy()
    test_out["label"] = y_test.values

    train_out.to_csv("splits_features_train.csv", index=False)
    val_out.to_csv("splits_features_val.csv", index=False)
    test_out.to_csv("splits_features_test.csv", index=False)
    print("\n✅ Saved: splits_features_train/val/test.csv")

    # -----------------------------
    # 2) Model A: Baseline (no imbalance handling)
    # -----------------------------
    print("\n==============================")
    print("A) BASELINE (no imbalance handling)")
    print("==============================")

    model_base = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model_base.fit(X_train, y_train)

    val_base = evaluate(model_base, X_val, y_val, name="VAL - BASELINE")

    # -----------------------------
    # 3) Model B: class_weight='balanced'  (طريقة 1)
    # -----------------------------
    print("\n==============================")
    print("B) CLASS_WEIGHT = 'balanced'")
    print("==============================")

    model_w = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )
    model_w.fit(X_train, y_train)

    val_w = evaluate(model_w, X_val, y_val, name="VAL - CLASS_WEIGHT")

    # -----------------------------
    # 4) Model C: SMOTE on TRAIN only (طريقة 2)
    # -----------------------------
    val_sm = None
    model_sm = None

    if IMBLEARN_OK:
        print("\n==============================")
        print("C) SMOTE (TRAIN ONLY)")
        print("==============================")

        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

        print_dist("TRAIN (after SMOTE)", y_train_sm)

        model_sm = RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model_sm.fit(X_train_sm, y_train_sm)

        val_sm = evaluate(model_sm, X_val, y_val, name="VAL - SMOTE")
    else:
        print("\n⚠️ imblearn مش منصّب عندك، SMOTE ما رح يشتغل.")
        print("إذا بدك SMOTE: نفّذي")
        print("pip install imbalanced-learn")

    # -----------------------------
    # 5) Choose best model by PR-AUC then evaluate on TEST
    # -----------------------------
    candidates = [
        ("BASELINE", model_base, val_base),
        ("CLASS_WEIGHT", model_w, val_w),
    ]
    if val_sm is not None:
        candidates.append(("SMOTE", model_sm, val_sm))

    # الأفضل حسب PR-AUC (لأنه أنسب للـ imbalance)
    best_name, best_model, best_scores = sorted(
        candidates, key=lambda x: x[2]["pr_auc"], reverse=True
    )[0]

    print("\n==============================")
    print(f"✅ BEST MODEL (by PR-AUC on VAL): {best_name}")
    print(f"VAL PR-AUC = {best_scores['pr_auc']:.6f} | VAL ROC-AUC = {best_scores['roc_auc']:.6f}")
    print("==============================")

    # Final Test evaluation
    evaluate(best_model, X_test, y_test, name=f"TEST - {best_name}")

if __name__ == "__main__":
    main()