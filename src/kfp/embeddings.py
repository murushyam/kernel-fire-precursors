"""Kernel mean embeddings and distribution regression (Szabo et al., 2016) via KRR."""
from __future__ import annotations
import numpy as np
from .mmd import rbf_kernel, median_heuristic


def mean_embedding_gram(bags_a, bags_b, bandwidth):
    """<mu_P, mu_Q> = mean_{x in P, y in Q} k(x,y) for every pair of bags."""
    G = np.empty((len(bags_a), len(bags_b)))
    for i, A in enumerate(bags_a):
        for j, B in enumerate(bags_b):
            G[i, j] = rbf_kernel(A, B, bandwidth).mean()
    return G


class DistributionRegressor:
    """Kernel ridge regression on bags of samples.

    kernel='linear' uses K = <mu_P, mu_Q>; kernel='gaussian' uses
    exp(-||mu_P - mu_Q||^2 / (2 gamma^2)) (second-level RBF)."""

    def __init__(self, bandwidth=None, alpha=1e-3, kernel="gaussian", gamma=None):
        self.bandwidth, self.alpha, self.kernel, self.gamma = bandwidth, alpha, kernel, gamma

    def _bags(self, bags):
        return [np.asarray(b, float).reshape(len(b), -1) for b in bags]

    def _outer(self, G, da, db):
        if self.kernel == "linear":
            return G
        D = da[:, None] + db[None, :] - 2 * G
        return np.exp(-np.maximum(D, 0) / (2 * self.gamma_ ** 2))

    def fit(self, bags, y):
        self.bags_ = self._bags(bags)
        self.bw_ = self.bandwidth or median_heuristic(np.vstack(self.bags_))
        G = mean_embedding_gram(self.bags_, self.bags_, self.bw_)
        self.d_ = np.diag(G).copy()
        if self.kernel != "linear":
            D = self.d_[:, None] + self.d_[None, :] - 2 * G
            self.gamma_ = self.gamma or np.sqrt(np.median(D[D > 0]) / 2 + 1e-12)
        K = self._outer(G, self.d_, self.d_)
        y = np.asarray(y, float); self.ymean_ = y.mean()
        self.coef_ = np.linalg.solve(K + self.alpha * len(y) * np.eye(len(y)), y - self.ymean_)
        return self

    def predict(self, bags):
        bags = self._bags(bags)
        G = mean_embedding_gram(bags, self.bags_, self.bw_)
        dq = np.array([rbf_kernel(b, b, self.bw_).mean() for b in bags])
        return self._outer(G, dq, self.d_) @ self.coef_ + self.ymean_
