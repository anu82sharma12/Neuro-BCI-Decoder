from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import eigh
import numpy as np

class CSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.cov_ = []
        for c in self.classes_:
            Xc = X[y == c]
            cov = np.mean([np.cov(x, rowvar=True) for x in Xc], axis=0)
            self.cov_.append(cov)
        eigvals, eigvecs = eigh(self.cov_[0], self.cov_[0] + self.cov_[1])
        idx = np.argsort(eigvals)[::-1]
        self.filters_ = eigvecs[:, idx[:self.n_components]]
        return self

    def transform(self, X):
        return np.array([self.filters_.T @ x for x in X])
