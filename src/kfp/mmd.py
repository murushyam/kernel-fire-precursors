"""Unbiased MMD^2 with RBF kernel and permutation test (Gretton et al., 2012)."""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist


def _2d(a):
    a = np.asarray(a, float)
    return a[:, None] if a.ndim == 1 else a


def median_heuristic(X, Y=None):
    Z = _2d(X) if Y is None else np.vstack([_2d(X), _2d(Y)])
    if len(Z) > 2000:
        Z = Z[np.random.default_rng(0).choice(len(Z), 2000, replace=False)]
    d = cdist(Z, Z, "euclidean")
    med = np.median(d[np.triu_indices_from(d, 1)])
    return float(med) if med > 0 else 1.0


def rbf_kernel(A, B, bandwidth):
    return np.exp(-cdist(_2d(A), _2d(B), "sqeuclidean") / (2.0 * bandwidth ** 2))


def _mmd2_from_K(K, m):
    Kxx, Kyy, Kxy = K[:m, :m], K[m:, m:], K[:m, m:]
    n = K.shape[0] - m
    return ((Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
            + (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
            - 2.0 * Kxy.mean())


def mmd2_unbiased(X, Y, bandwidth=None):
    X, Y = _2d(X), _2d(Y)
    if len(X) < 2 or len(Y) < 2:
        raise ValueError("need >=2 samples per group")
    bw = bandwidth or median_heuristic(X, Y)
    Z = np.vstack([X, Y])
    return float(_mmd2_from_K(rbf_kernel(Z, Z, bw), len(X)))


def mmd_permutation_test(X, Y, bandwidth=None, n_perm=500, rng=None):
    """Return (mmd2, p_value) under H0: P_X = P_Y."""
    X, Y = _2d(X), _2d(Y)
    rng = np.random.default_rng(rng)
    bw = bandwidth or median_heuristic(X, Y)
    Z = np.vstack([X, Y]); m = len(X)
    K = rbf_kernel(Z, Z, bw)
    stat = _mmd2_from_K(K, m)
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(len(Z))
        null[i] = _mmd2_from_K(K[np.ix_(p, p)], m)
    pval = (1 + np.sum(null >= stat)) / (1 + n_perm)
    return float(stat), float(pval)
