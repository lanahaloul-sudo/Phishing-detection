import os
import re
import json
import math
from urllib.parse import urlparse, unquote

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# =========================
# Optional SHAP (Tabular only)
# =========================
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# =========================
# UI Style
# =========================
def inject_css():
    css = """
    <style>
    .stApp{
      background: radial-gradient(circle at 15% 20%, rgba(0,180,255,0.18), transparent 40%),
                  radial-gradient(circle at 85% 25%, rgba(255,0,120,0.12), transparent 45%),
                  radial-gradient(circle at 30% 85%, rgba(0,255,140,0.10), transparent 45%),
                  linear-gradient(135deg, #0b0f17 0%, #060913 60%, #050812 100%);
      color: #e8eefc;
    }
    .glass {
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 16px;
      padding: 14px 16px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    }
    .title {
      font-size: 42px;
      font-weight: 800;
      letter-spacing: 0.2px;
      margin-bottom: 6px;
      background: linear-gradient(90deg, #7cf7ff 0%, #a78bfa 50%, #ff72b6 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .subtitle { opacity: 0.85; margin-top: -6px; }
    .small { opacity: 0.8; font-size: 12px; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# =========================
# Paths (YOUR project)
# =========================
RAW_MODEL_PATH = "model_raw_new.joblib"
TAB_MODEL_PATH = "model_tabular_clean.joblib"
HYB_MODEL_PATH = os.path.join("results_hybrid", "model_hybrid.joblib")


# =========================
# Utils
# =========================
def load_joblib(path):
    if not os.path.exists(path):
        return None, f"Missing file: {path}"
    try:
        return joblib.load(path), None
    except Exception as e:
        return None, f"Failed to load {path}: {repr(e)}"


def normalize_url(url: str) -> str:
    url = (url or "").strip()
    url = unquote(url)
    if url and not re.match(r"^[a-zA-Z]+://", url):
        url = "http://" + url
    return url


def pick_url_col(df: pd.DataFrame):
    """Try to detect URL column name from common candidates."""
    if df is None or df.empty:
        return None
    candidates = ["url_raw", "url", "URL", "Url", "link", "uri"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: first column that looks like url-ish
    for c in df.columns:
        if "url" in c.lower():
            return c
    return df.columns[0]


def safe_predict_proba_phish(model, X: pd.DataFrame):
    """
    Returns phishing probability.
    Your labeling in project might be:
      - 1 = phishing, 0 = benign  (common)
      - or 0 = phishing, 1 = benign (you wrote this sometimes)
    We handle BOTH robustly by checking model.classes_ if available.
    """
    if model is None or X is None or len(X) == 0:
        return None, "No model or empty input."

    if not hasattr(model, "predict_proba"):
        return None, "Model has no predict_proba."

    proba = model.predict_proba(X)
    # find which column is phishing
    phish_index = None
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        # prefer phishing=1
        if 1 in classes:
            phish_index = classes.index(1)
        elif 0 in classes:
            # fallback phishing=0
            phish_index = classes.index(0)
        else:
            phish_index = 0
    else:
        phish_index = 0

    p = float(proba[0][phish_index])
    return p, None


def unwrap_pipeline(model):
    """Return (preprocess, estimator). preprocess may be None."""
    if hasattr(model, "steps"):
        return model[:-1], model.steps[-1][1]
    return None, model

def get_feature_names_from_preprocess(preprocess):
    if preprocess is None:
        return None
    try:
        return list(preprocess.get_feature_names_out())
    except Exception:
        return None


def explain_linear_pipeline_local(model, X_df, top_k=25):
    """
    Local explanation for HYBRID pipeline (linear).
    contribution = x_i * coef_i
    """
    preprocess, clf = unwrap_pipeline(model)

    if not hasattr(clf, "coef_"):
        return None, "Estimator has no coef_ (not linear)."

    # transform
    try:
        Xt = preprocess.transform(X_df) if preprocess is not None else X_df.values
    except Exception as e:
        return None, f"Transform failed: {repr(e)}"

    fn = get_feature_names_from_preprocess(preprocess)
    if fn is None:
        fn = [f"f{i}" for i in range(Xt.shape[1])]

    coef = clf.coef_
    if coef.ndim == 2:
        coef = coef[0]
    coef = np.asarray(coef).ravel()

    # contributions (sparse-safe)
    try:
        if hasattr(Xt, "multiply"):
            contrib = Xt.multiply(coef).toarray().ravel()
            xvals = Xt.toarray().ravel()
        else:
            contrib = (Xt.ravel() * coef)
            xvals = Xt.ravel()
    except Exception as e:
        return None, f"Contribution compute failed: {repr(e)}"

    idx = np.argsort(np.abs(contrib))[::-1][:top_k]
    out = pd.DataFrame({
        "feature": [fn[i] if i < len(fn) else f"f{i}" for i in idx],
        "value": [float(xvals[i]) for i in idx],
        "coef": [float(coef[i]) for i in idx],
        "contribution": [float(contrib[i]) for i in idx],
    })
    out["effect"] = out["contribution"].apply(lambda v: "↑ phishing" if v > 0 else "↓ phishing")
    return out, None


def shap_tabular_pipeline_local(model, X_row, top_k=15):
    """
    SHAP only for Tabular (small features).
    """
    if not SHAP_AVAILABLE:
        return None, "SHAP not installed. Run: pip install shap"

    preprocess, clf = unwrap_pipeline(model)

    # transform
    try:
        Xt = preprocess.transform(X_row) if preprocess is not None else X_row.values
    except Exception as e:
        return None, f"Transform failed: {repr(e)}"

    fn = get_feature_names_from_preprocess(preprocess)
    if fn is None:
        fn = list(X_row.columns)

    # try linear explainer
    try:
        explainer = shap.LinearExplainer(clf, Xt, feature_perturbation="interventional")
        sv = explainer.shap_values(Xt)
    except Exception as e:
        return None, f"SHAP failed: {repr(e)}"

    if isinstance(sv, list):
        sv = sv[0]
    sv = np.array(sv).ravel()

    idx = np.argsort(np.abs(sv))[::-1][:top_k]
    df = pd.DataFrame({
        "feature": [fn[i] if i < len(fn) else f"f{i}" for i in idx],
        "shap": [float(sv[i]) for i in idx],
    })
    return df, None


# =========================
# Build X for each model
# =========================
def build_X_raw(url: str):
    # your raw model usually expects df with column "url_raw" or "url"
    return pd.DataFrame({"url_raw": [url], "url": [url]})


def build_X_tabular(url: str):
    """
    Your tabular model is a CLEAN pipeline in model_tabular_clean.joblib
    It may include its own feature engineering or expects numeric columns.
    If your tabular pipeline expects URL column, this will work.
    If it expects numeric-only already, then you should feed the numeric vector.
    We'll try URL-based first because your project uses URL-based features.
    """
    return pd.DataFrame({"url_raw": [url], "url": [url]})


def build_X_hybrid(url: str):
    """
    Your hybrid pipeline in train_hybrid.py uses HybridFeatures(url_col='url_raw')
    So it expects a DF containing that column.
    """
    return pd.DataFrame({"url_raw": [url]})


# =========================
# MAIN APP
# =========================
def main():
    st.set_page_config(page_title="Phishing Detection", page_icon="🛡", layout="wide")
    inject_css()

    st.markdown("<div class='title'>🛡 Phishing Detection</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Supports your trained models: <b>RAW</b>, <b>TABULAR</b>, <b>HYBRID</b>. "
        "Includes explanations: <b>HYBRID local contributions</b> + <b>SHAP for TABULAR</b> (optional).</div>",
        unsafe_allow_html=True
    )
    st.write("")

    # Load models
    raw_model, raw_err = load_joblib(RAW_MODEL_PATH)
    tab_model, tab_err = load_joblib(TAB_MODEL_PATH)
    hyb_model, hyb_err = load_joblib(HYB_MODEL_PATH)

    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.write(f"RAW model: {'✅ Loaded' if raw_model else '⚠️ Not loaded'}")
    if raw_err and not raw_model:
        st.caption(raw_err)
    st.write(f"TABULAR model: {'✅ Loaded' if tab_model else '⚠️ Not loaded'}")
    if tab_err and not tab_model:
        st.caption(tab_err)
    st.write(f"HYBRID model: {'✅ Loaded' if hyb_model else '⚠️ Not loaded'}")
    if hyb_err and not hyb_model:
        st.caption(hyb_err)
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # Controls
    colA, colB, colC = st.columns([1.1, 1.1, 1.8])
    with colA:
        mode = st.selectbox(
            "Mode",
            ["RAW", "TABULAR", "HYBRID"],
            index=2 if hyb_model else (1 if tab_model else 0),
        )
    with colB:
        threshold = st.slider("Decision threshold (phishing)", 0.01, 0.99, 0.50, 0.01)
        show_explain = st.checkbox("Show explanation", value=True)
    with colC:
        url = st.text_input(
            "Paste URL here",
            value="http://secure-login.verify.account-update.paypal.example.net/login.php?user=check&token=88772342"
        )

    c1, c2 = st.columns([1, 1])
    do_predict = c1.button("Predict", use_container_width=True)
    do_test = c2.button("Run test suite", use_container_width=True)

    if do_test:
        tests = [
            "https://www.google.com",
            "https://tranco-list.eu",
            "http://account.verify.microsoft.security.example/login",
            "http://192.168.1.45@secure-verify-login.example.com/update/account",
        ]
        st.subheader("Test suite")
        for t in tests:
            st.write("—", t)
        st.info("Press Predict with any URL to see output.")

    if not do_predict:
        return

    url = normalize_url(url)

    # Choose model + build X
    chosen_model = None
    X = None
    if mode == "RAW":
        chosen_model = raw_model
        X = build_X_raw(url)
    elif mode == "TABULAR":
        chosen_model = tab_model
        X = build_X_tabular(url)
    else:
        chosen_model = hyb_model
        X = build_X_hybrid(url)

    if chosen_model is None:
        st.error(f"{mode} model is not loaded.")
        return

    # Predict
    p_phish, perr = safe_predict_proba_phish(chosen_model, X)
    if perr:
        st.error(f"Prediction failed: {perr}")
        with st.expander("Debug: X used for prediction"):
            st.dataframe(X.T, use_container_width=True)
        return

    is_phish = (p_phish >= threshold)

    if is_phish:
        st.error(f"Prediction: PHISHING | P(phish) = {p_phish:.4f}")
    else:
        st.success(f"Prediction: BENIGN | P(phish) = {p_phish:.4f}")

    st.caption(f"Threshold = {threshold:.2f}")

    # =========================
    # Debug: what columns the model expects
    # =========================
    with st.expander("Debug: Model info"):
        preprocess, clf = unwrap_pipeline(chosen_model)
        st.write("Model type:", type(chosen_model).name)
        st.write("Estimator:", type(clf).name)
        if hasattr(chosen_model, "classes_"):
            st.write("Classes:", list(chosen_model.classes_))
        fn = get_feature_names_from_preprocess(preprocess)
        st.write("Has get_feature_names_out:", fn is not None)
        if fn is not None:
            st.write("Example feature names:", fn[:20])

    with st.expander("Debug: X used for prediction (first row)"):
        st.dataframe(X.head(1).T.rename(columns={0: "value"}), use_container_width=True)

    # =========================
    # Explanation
    # =========================
    if show_explain:
        st.subheader("Explanation")

        if mode == "HYBRID":
            expl_df, expl_err = explain_linear_pipeline_local(chosen_model, X, top_k=25)
            if expl_err:
                st.warning("HYBRID explanation not available: " + expl_err)
                st.info("If HYBRID is not linear, I can switch to another explanation method.")
            else:
                st.caption("Top local contributions (x_i * coef_i). Positive → pushes toward phishing.")
                st.dataframe(expl_df, use_container_width=True)

        elif mode == "TABULAR":
            # SHAP only for tabular
            if not SHAP_AVAILABLE:
                st.warning("SHAP not installed. Run: pip install shap")
            else:
                # Try SHAP on pipeline (may still fail if model is not linear)
                shap_df, shap_err = shap_tabular_pipeline_local(chosen_model, X, top_k=15)
                if shap_err:
                    st.warning(shap_err)
                else:
                    st.caption("Top SHAP features (Tabular).")
                    st.dataframe(shap_df, use_container_width=True)

        else:
            st.info("RAW explanation can be heavy. Use HYBRID mode for best interpretable results.")


main()