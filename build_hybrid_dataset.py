import pandas as pd

RAW_PATH = "final_raw_dataset.csv"
FEAT_PATH = "final_features_dataset.csv"
OUT_PATH = "hybrid_dataset.csv"

def main():
    raw = pd.read_csv(RAW_PATH)
    feat = pd.read_csv(FEAT_PATH)

    # تأكيد وجود الأعمدة
    assert "label" in raw.columns
    assert "label" in feat.columns
    assert "url_raw" in raw.columns

    # تأكيد نفس الطول
    if len(raw) != len(feat):
        raise ValueError(f"Length mismatch: raw={len(raw)}, feat={len(feat)}")

    # حذف label من features حتى ما يتكرر
    feat_no_label = feat.drop(columns=["label"])

    # دمج
    hybrid = pd.concat(
        [raw[["url_raw", "label"]], feat_no_label],
        axis=1
    )

    hybrid.to_csv(OUT_PATH, index=False)
    print("✅ Hybrid dataset saved:", OUT_PATH)
    print("Columns:", hybrid.columns.tolist())

if __name__ == "__main__":
    main()