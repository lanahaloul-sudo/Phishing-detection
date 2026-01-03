import pandas as pd
import numpy as np
from urllib.parse import urlparse
import math
import re

# -----------------------------
# Feature functions
# -----------------------------
def entropy(s):
    if not s:
        return 0
    prob = [float(s.count(c)) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in prob)

def extract_features(url):
    parsed = urlparse(url)

    features = {}
    features["url_length"] = len(url)
    features["num_digits"] = sum(c.isdigit() for c in url)
    features["num_special"] = sum(not c.isalnum() for c in url)
    features["has_https"] = int(parsed.scheme == "https")
    features["num_subdomains"] = parsed.netloc.count(".")
    features["path_length"] = len(parsed.path)
    features["entropy"] = entropy(url)
    features["has_ip"] = int(bool(re.search(r"\d+\.\d+\.\d+\.\d+", url)))

    return features

# -----------------------------
# Main
# -----------------------------
def main():
    INPUT_FILE = "final_dataset_clean.csv"
    OUTPUT_FILE = "features_dataset.csv"

    df = pd.read_csv(INPUT_FILE)

    feature_rows = []
    for url in df["url_raw"]:
        feature_rows.append(extract_features(url))

    features_df = pd.DataFrame(feature_rows)
    features_df["label"] = df["label"].values

    features_df.to_csv(OUTPUT_FILE, index=False)

    print("✅ Features extracted successfully")
    print(features_df.head())
    print(features_df["label"].value_counts())

if __name__ == "__main__":
    main()