import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

# =========================
# Paths 
# =========================
RAW_TEST_PATH = "splits/test.csv"
TAB_TEST_PATH = "splits_tabular_test.csv"          
HYB_TEST_PATH = "splits_hybrid/test.csv"           

RAW_MODEL_PATH = "model_raw_new.joblib"
TAB_MODEL_PATH = "model_tabular_clean.joblib"
HYB_MODEL_PATH = "results_hybrid/model_hybrid.joblib"

OUT_DIR = "results_compare"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# Helpers
# =========================
def pick_col(df, candidates):
    """Pick first existing column name from candidates (case-insensitive)."""
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower().strip() in cols_lower:
            return cols_lower[c.lower().strip()]
    return None

def load_if_exists(path):
    if path and os.path.exists(path):
        return joblib.load(path)
    return None

def safe_predict_proba(model, X):
    """Return proba for positive class if available, else None."""
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1]
        # أحياناً بيرجع عمود واحد
        return p.ravel()
    return None

def ensure_raw_input(X):
    """
    RAW model غالباً مدرّب على عمود واحد (url string).
    إذا DataFrame عمود واحد -> نحولو Series.
    """
    if isinstance(X, pd.DataFrame):
        if X.shape[1] != 1:
            # إذا صار عندك اكتر من عمود بالغلط، خدي أول عمود بس
            return X.iloc[:, 0]
        return X.iloc[:, 0]
    return X

def align_hybrid_columns_for_model(hyb_model, X_hyb):
    """
    يحاول يطابق أعمدة الـ Hybrid حسب ما كان متعلم وقت التدريب:
    - إذا لقى numeric_cols_ داخل step اسمه 'hyb' => بياخد url_raw + هالأعمدة فقط
    - إذا ما لقى => بياخد فقط url_raw (يعني تدرب URL-only hybrid)
    """
    if hyb_model is None or X_hyb is None:
        return X_hyb

    # لازم يكون عندنا url column
    if "url_raw" not in X_hyb.columns:
        # جرّبي أسماء بديلة
        url_col = pick_col(X_hyb, ["url_raw", "url", "URL", "Url"])
        if url_col is None:
            raise ValueError("Hybrid test data must contain url_raw (or url) column.")
        if url_col != "url_raw":
            X_hyb = X_hyb.rename(columns={url_col: "url_raw"})

    hyb_step = None
    if hasattr(hyb_model, "named_steps") and "hyb" in hyb_model.named_steps:
        hyb_step = hyb_model.named_steps["hyb"]

    if hyb_step is not None and hasattr(hyb_step, "numeric_cols_"):
        keep_numeric = [c for c in hyb_step.numeric_cols_ if c in X_hyb.columns]
        keep = ["url_raw"] + keep_numeric
        X_hyb = X_hyb[keep].copy()
    else:
        # ما في numeric_cols_ محفوظة -> غالباً تدرب بدون numeric
        X_hyb = X_hyb[["url_raw"]].copy()

    return X_hyb

def plot_and_save_cm(cm, title, out_path):
    fig = plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

def render_table_image(df, out_path, title="Final Comparison Table"):
    fig = plt.figure(figsize=(12, 0.5 + 0.35 * (len(df) + 1)))
    plt.axis("off")
    plt.title(title)
    table = plt.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def evaluate_model(name, model, X, y, is_raw=False, save_prefix=None):
    if model is None:
        print(f"[WARN] {name} model not found. Skipping.")
        return None

    if is_raw:
        X_in = ensure_raw_input(X)
    else:
        X_in = X

    pred = model.predict(X_in)
    proba = safe_predict_proba(model, X_in)

    cm = confusion_matrix(y, pred)
    report = classification_report(y, pred, digits=4)

    acc = accuracy_score(y, pred)
    prec = precision_score(y, pred, zero_division=0)
    rec = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)

    roc = roc_auc_score(y, proba) if proba is not None else np.nan
    pr = average_precision_score(y, proba) if proba is not None else np.nan

    print(f"\n===== {name} =====")
    print(report)
    print("Confusion Matrix:\n", cm)
    print(f"ACC={acc:.6f} | P={prec:.6f} | R={rec:.6f} | F1={f1:.6f} | ROC-AUC={roc:.6f} | PR-AUC={pr:.6f}")

    # Save confusion matrix image
    if save_prefix:
        cm_path = os.path.join(OUT_DIR, f"{save_prefix}_cm.png")
        plot_and_save_cm(cm, f"{name} - Confusion Matrix", cm_path)

        # Save metrics json
        metrics_path = os.path.join(OUT_DIR, f"{save_prefix}_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": name,
                    "accuracy": float(acc),
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1": float(f1),
                    "roc_auc": None if np.isnan(roc) else float(roc),
                    "pr_auc": None if np.isnan(pr) else float(pr),
                    "confusion_matrix": cm.tolist(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC_AUC": roc,
        "PR_AUC": pr,
        "TN": int(cm[0, 0]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "TP": int(cm[1, 1]),
    }

# =========================
# Main
# =========================
def main():
    # Load models
    raw_model = load_if_exists(RAW_MODEL_PATH)
    tab_model = load_if_exists(TAB_MODEL_PATH)
    hyb_model = load_if_exists(HYB_MODEL_PATH)

    # ---------- RAW data ----------
    raw_df = pd.read_csv(RAW_TEST_PATH)
    raw_url_col = pick_col(raw_df, ["url_raw", "url", "URL", "Url"])
    raw_label_col = pick_col(raw_df, ["label", "Label", "y", "target"])

    if raw_url_col is None or raw_label_col is None:
        raise ValueError("RAW test must contain url column + label column.")

    X_raw = raw_df[[raw_url_col]].rename(columns={raw_url_col: "url_raw"})
    y_raw = raw_df[raw_label_col].astype(int).values

    # ---------- TABULAR data ----------
    tab_df = pd.read_csv(TAB_TEST_PATH)
    tab_label_col = pick_col(tab_df, ["label", "Label", "y", "target"])
    if tab_label_col is None:
        raise ValueError("TABULAR test must contain label column.")

    X_tab = tab_df.drop(columns=[tab_label_col])
    y_tab = tab_df[tab_label_col].astype(int).values

    # ---------- HYBRID data ----------
    X_hyb = y_hyb = None
    if hyb_model is not None:
        if HYB_TEST_PATH and os.path.exists(HYB_TEST_PATH):
            hyb_df = pd.read_csv(HYB_TEST_PATH)
            hyb_label_col = pick_col(hyb_df, ["label", "Label", "y", "target"])
            if hyb_label_col is None:
                raise ValueError("HYBRID test must contain label column.")

            y_hyb = hyb_df[hyb_label_col].astype(int).values
            X_hyb = hyb_df.drop(columns=[hyb_label_col])

        else:
            # إذا ما عندك HYB test جاهز: جرّب ندمج raw + tabular إذا نفس الطول

            if len(raw_df) == len(tab_df):
                print("[INFO] HYB_TEST_PATH not found. Building hybrid from RAW + TABULAR test (same length).")
                # url_raw + numeric features
                url_part = raw_df[[raw_url_col]].rename(columns={raw_url_col: "url_raw"})
                tab_part = tab_df.drop(columns=[tab_label_col]).copy()
                X_hyb = pd.concat([url_part, tab_part], axis=1)
                y_hyb = y_raw
            else:
                print("[WARN] Can't build hybrid: raw and tabular tests have different lengths.")
                X_hyb = y_hyb = None

    # Align hybrid columns with the saved model expectations
    if X_hyb is not None and hyb_model is not None:
        X_hyb = align_hybrid_columns_for_model(hyb_model, X_hyb)

    # ---------- Evaluate ----------
    results = []

    r1 = evaluate_model("RAW", raw_model, X_raw, y_raw, is_raw=True, save_prefix="raw_test")
    if r1: results.append(r1)

    r2 = evaluate_model("TABULAR", tab_model, X_tab, y_tab, is_raw=False, save_prefix="tabular_test")
    if r2: results.append(r2)

    if hyb_model is not None and X_hyb is not None and y_hyb is not None:
        r3 = evaluate_model("HYBRID", hyb_model, X_hyb, y_hyb, is_raw=False, save_prefix="hybrid_test")
        if r3: results.append(r3)
    else:
        print("[WARN] HYBRID evaluation skipped (model or data missing).")

    # ---------- Final table ----------
    if not results:
        print("[ERROR] No results to save.")
        return

    df_res = pd.DataFrame(results)
    # ترتيب أعمدة لطيف
    col_order = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC", "TN", "FP", "FN", "TP"]
    df_res = df_res[col_order]

    csv_path = os.path.join(OUT_DIR, "final_comparison_table.csv")
    df_res.to_csv(csv_path, index=False)

    png_path = os.path.join(OUT_DIR, "final_comparison_table.png")
    render_table_image(df_res.round(6), png_path)

    print("\n✅ Done!")
    print("Confusion matrices + metrics saved in:", OUT_DIR)
    print("Table CSV :", csv_path)
    print("Table PNG :", png_path)

if __name__ == "__main__":
    main()