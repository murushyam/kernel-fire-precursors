import numpy as np
from kfp.embeddings import DistributionRegressor

def _bags(n, r):
    mus = r.uniform(-2, 2, n)
    return [r.normal(m, 1, size=(30, 1)) for m in mus], mus

def test_distribution_regression_recovers_mean():
    r = np.random.default_rng(0)
    btr, ytr = _bags(80, r); bte, yte = _bags(30, r)
    pred = DistributionRegressor(alpha=1e-4).fit(btr, ytr).predict(bte)
    assert np.corrcoef(pred, yte)[0, 1] > 0.9

def test_linear_kernel_runs():
    r = np.random.default_rng(1); b, y = _bags(20, r)
    assert DistributionRegressor(kernel="linear").fit(b, y).predict(b).shape == (20,)
