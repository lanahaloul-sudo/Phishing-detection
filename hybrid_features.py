from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import numpy as np

class HybridFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, url_col="url_raw", ngram_range=(3,6), min_df=2, max_features=250000):
        self.url_col = url_col
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=True
        )

    def fit(self, X, y=None):
        self.vectorizer.fit(X[self.url_col])
        return self

    def transform(self, X):
        X_text = self.vectorizer.transform(X[self.url_col])
        X_tab = X.drop(columns=[self.url_col]).values
        return hstack([X_text, X_tab])