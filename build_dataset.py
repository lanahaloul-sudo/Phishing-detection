import pandas as pd
from urllib.parse import urlparse
import re

# -----------------------------
# 1) Helpers: choose URL column, normalize, canonicalize
# -----------------------------

URL_COL_CANDIDATES = [
    "url", "URL", "Url", "Url_raw", "url_raw", "URL_raw",
    "urlraw", "UrlRaw", "URLRaw"
]

def pick_url_col(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    for c in URL_COL_CANDIDATES:
        if c in cols:
            return c
    # Try fuzzy: strip spaces/lower
    lowered = {c.lower().strip(): c for c in cols}
    for c in [x.lower() for x in URL_COL_CANDIDATES]:
        if c in lowered:
            return lowered[c]
    raise KeyError(f"Could not find a URL column. Available columns: {cols[:30]}...")

def ensure_scheme(u: str) -> str:
    u = str(u).strip()
    if not u:
        return u
    if not u.startswith(("http://", "https://")):
        return "http://" + u
    return u

def canonical_url(u: str) -> str:
    """
    Canonical form used for dedup + split safety:
    - lowercase
    - ensure scheme (so urlparse works)
    - remove leading www.
    - remove default ports
    - strip trailing slash
    - keep host + path only (drop query/fragment)
    """
    u = str(u).strip()
    if not u:
        return ""

    u = u.strip().lower()
    u = ensure_scheme(u)

    # Remove whitespace inside (rare but happens)
    u = re.sub(r"\s+", "", u)

    p = urlparse(u)

    host = p.netloc
    # drop credentials if present: user:pass@host
    if "@" in host:
        host = host.split("@", 1)[1]

    # remove www.
    if host.startswith("www."):
        host = host[4:]

    # remove default ports
    host = host.replace(":80", "").replace(":443", "")

    path = (p.path or "").rstrip("/")

    return f"{host}{path}"

def clean_one_source(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    url_col = pick_url_col(df)
    out = df[[url_col]].copy()
    out = out.rename(columns={url_col: "url_raw"})
    out["url_raw"] = out["url_raw"].astype(str).str.strip()
    out = out[out["url_raw"].notna() & (out["url_raw"] != "")]

    # canonical key
    out["url_canon"] = out["url_raw"].apply(canonical_url)

    # drop empty canon
    out = out[out["url_canon"].notna() & (out["url_canon"] != "")]

    # dedup within the same source
    before = len(out)
    out = out.drop_duplicates(subset=["url_canon"]).reset_index(drop=True)
    after = len(out)

    print(f"[{source_name}] URL column used: {url_col}")
    print(f"[{source_name}] rows before={before}, after dedup={after}, removed={before-after}")
    return out


# -----------------------------
# 2) Main: load, clean, dedup, remove overlaps, save
# -----------------------------
def main():
    # عدلي أسماء الملفات إذا مختلفة
    PHISHTANK_IN = "phishtank.csv"
    TRANCO_IN = "tranco_urls.csv"

    # مخرجات نظيفة
    OUT_PHISH_CLEAN = "phishtank_clean.csv"
    OUT_BENIGN_CLEAN = "tranco_clean.csv"

    # نسخة بلا أي تداخل بين المصدرين (اختياري لكن مفيد)
    OUT_PHISH_NO_OVERLAP = "phishtank_clean_no_overlap.csv"
    OUT_BENIGN_NO_OVERLAP = "tranco_clean_no_overlap.csv"

    # load
    ph_df = pd.read_csv(PHISHTANK_IN)
    tr_df = pd.read_csv(TRANCO_IN)

    # clean + dedup داخل كل مصدر
    ph = clean_one_source(ph_df, "PhishTank")
    tr = clean_one_source(tr_df, "Tranco")

    # احصائية تداخل بين المصدرين (نفس url_canon موجود بمصدرين)
    overlap = set(ph["url_canon"]).intersection(set(tr["url_canon"]))
    print(f"[Overlap] URLs appearing in BOTH PhishTank & Tranco = {len(overlap)}")

    # نحذف التداخل من Tranco (أو من الاثنين، حسب قرارك)
    # الأفضل غالباً: نخلي PhishTank مثل ما هو، ونشيل المتداخل من benign حتى ما يصير label conflict
    tr_no_overlap = tr[~tr["url_canon"].isin(overlap)].reset_index(drop=True)
    ph_no_overlap = ph.copy()

    # حفظ الملفات
    ph.to_csv(OUT_PHISH_CLEAN, index=False)
    tr.to_csv(OUT_BENIGN_CLEAN, index=False)
    ph_no_overlap.to_csv(OUT_PHISH_NO_OVERLAP, index=False)
    tr_no_overlap.to_csv(OUT_BENIGN_NO_OVERLAP, index=False)
    print(f"Saved -> {OUT_PHISH_CLEAN}")
    print(f"Saved -> {OUT_BENIGN_CLEAN}")
    print(f"Saved -> {OUT_PHISH_NO_OVERLAP}")
    print(f"Saved -> {OUT_BENIGN_NO_OVERLAP}")

    # ملاحظة: لسا ما دمجنا. الدمج بنعمله بمرحلة لاحقة مثل ما طلبتي.

if __name__ == "__main__":
    main()