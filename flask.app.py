
from flask import Flask, request, render_template_string
import joblib
import os
import pandas as pd
import importlib

app = Flask(__name__)

# ========= Paths =========
RAW_MODEL_PATH = "model_raw_new.joblib"
TAB_MODEL_PATH = "model_tabular_clean.joblib"
HYB_MODEL_PATH = "results_hybrid/model_hybrid.joblib"

# ========= Load models safely =========
raw_model = joblib.load(RAW_MODEL_PATH) if os.path.exists(RAW_MODEL_PATH) else None
tab_model = joblib.load(TAB_MODEL_PATH) if os.path.exists(TAB_MODEL_PATH) else None
hyb_model = joblib.load(HYB_MODEL_PATH) if os.path.exists(HYB_MODEL_PATH) else None


# ========= Helper: find your function automatically =========
def find_callable(module_name: str, candidates: list[str]):
    mod = importlib.import_module(module_name)
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    return None


# Try to locate your tabular extractor in url_feature_extractor.py
EXTRACT_TAB_FN = None
try:
    EXTRACT_TAB_FN = find_callable(
        "url_feature_extractor",
        [
            "extract_features_single",
            "extract_features",
            "extract_url_features",
            "url_to_features",
            "compute_features",
            "make_features",
        ],
    )
except Exception:
    EXTRACT_TAB_FN = None

# Try to locate your hybrid builder in hybrid_features.py
BUILD_HYB_FN = None
try:
    BUILD_HYB_FN = find_callable(
        "hybrid_features",
        [
            "build_hybrid_single",
            "build_hybrid_features",
            "make_hybrid_single",
            "url_to_hybrid",
            "build_single",
        ],
    )
except Exception:
    BUILD_HYB_FN = None


def ensure_url_series(url: str):
    return pd.Series([url])


def to_dataframe_one_row(obj):
    """accept dict/Series/DataFrame -> returns DataFrame with 1 row"""
    if isinstance(obj, dict):
        return pd.DataFrame([obj])
    if isinstance(obj, pd.Series):
        return obj.to_frame().T
    if isinstance(obj, pd.DataFrame):
        return obj
    raise TypeError(f"Unsupported feature output type: {type(obj)}")


def predict_with_model(model, X):
    pred = int(model.predict(X)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0][1])
    return pred, proba


def predict_raw(url: str):
    X = ensure_url_series(url)
    return predict_with_model(raw_model, X)


def predict_tabular(url: str):
    if tab_model is None:
        raise ValueError("TABULAR model file not found.")
    if EXTRACT_TAB_FN is None:
        raise ValueError("Tabular extractor function not found in url_feature_extractor.py")
    feats = EXTRACT_TAB_FN(url)
    X = to_dataframe_one_row(feats)
    return predict_with_model(tab_model, X)


def predict_hybrid(url: str):
    if hyb_model is None:
        raise ValueError("HYBRID model file not found.")
    if BUILD_HYB_FN is None:
        raise ValueError("Hybrid builder function not found in hybrid_features.py")
    feats = BUILD_HYB_FN(url)
    X = to_dataframe_one_row(feats)
    return predict_with_model(hyb_model, X)


HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Phishing Detection</title>
  <style>
    body{font-family:Arial; background:#f6f7fb; margin:0; padding:0;}
    .wrap{max-width:800px; margin:40px auto; background:white; padding:24px; border-radius:16px; box-shadow:0 8px 30px rgba(0,0,0,.08);}
    h1{margin:0 0 12px 0;}
    .row{display:flex; gap:12px; flex-wrap:wrap;}
    input, select, button{padding:12px; border-radius:10px; border:1px solid #ddd; font-size:16px;}
    input{flex:1; min-width:260px;}
    button{cursor:pointer; border:none; background:#222; color:white;}
    .box{margin-top:16px; padding:14px; border-radius:12px; background:#f1f5ff;}
    .err{background:#ffecec; color:#8b0000;}
    small{color:#666;}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Phishing Detection</h1>
    <small>Models: RAW / TABULAR / HYBRID</small>

<form method="POST">
      <div class="row" style="margin-top:14px;">
        <input name="url" placeholder="Paste URL here..." value="{{url|default('')}}" required>
        <select name="model_type">
          <option value="RAW" {% if model_type=='RAW' %}selected{% endif %}>RAW</option>
          <option value="TABULAR" {% if model_type=='TABULAR' %}selected{% endif %}>TABULAR</option>
          <option value="HYBRID" {% if model_type=='HYBRID' %}selected{% endif %}>HYBRID</option>
        </select>
        <button type="submit">Predict</button>
      </div>
    </form>

    {% if error %}
      <div class="box err"><b>Error:</b> {{error}}</div>
    {% endif %}

    {% if result %}
      <div class="box">
        <div><b>Result:</b> {{result}}</div>
        {% if proba is not none %}
          <div><b>Phishing probability:</b> {{ "%.4f"|format(proba) }}</div>
        {% endif %}
      </div>
    {% endif %}

    <div style="margin-top:14px;">
      <small>
        Loaded models:
        RAW={{ 'OK' if raw_ok else 'Missing' }},
        TABULAR={{ 'OK' if tab_ok else 'Missing' }},
        HYBRID={{ 'OK' if hyb_ok else 'Missing' }}.
      </small>
    </div>
  </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    url = ""
    model_type = "RAW"
    result = None
    proba = None
    error = None

    if request.method == "POST":
        url = request.form.get("url", "").strip()
        model_type = request.form.get("model_type", "RAW")

        try:
            if model_type == "RAW":
                if raw_model is None:
                    raise ValueError("RAW model not found: model_raw_new.joblib")
                pred, proba = predict_raw(url)

            elif model_type == "TABULAR":
                pred, proba = predict_tabular(url)

            elif model_type == "HYBRID":
                pred, proba = predict_hybrid(url)

            else:
                raise ValueError("Unknown model type.")

            result = "PHISHING 🚨" if pred == 1 else "BENIGN ✅"

        except Exception as e:
            error = str(e)

    return render_template_string(
        HTML,
        url=url,
        model_type=model_type,
        result=result,
        proba=proba,
        error=error,
        raw_ok=(raw_model is not None),
        tab_ok=(tab_model is not None),
        hyb_ok=(hyb_model is not None),
    )


@app.route("/favicon.ico")
def favicon():
    # منع 404 المزعجة
    return ("", 204)


if __name__ == "__main__":
    app.run(debug=True)