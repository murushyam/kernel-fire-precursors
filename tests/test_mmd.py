import numpy as np
from kfp.mmd import mmd2_unbiased, mmd_permutation_test
from kfp.changepoint import mmd_scan

def test_same_distribution_not_rejected():
    r = np.random.default_rng(1); X, Y = r.normal(size=(150, 2)), r.normal(size=(150, 2))
    s, p = mmd_permutation_test(X, Y, n_perm=200, rng=0)
    assert abs(s) < 0.05 and p > 0.05

def test_shift_detected():
    r = np.random.default_rng(2); X, Y = r.normal(size=(150, 2)), r.normal(0.8, 1, size=(150, 2))
    s, p = mmd_permutation_test(X, Y, n_perm=200, rng=0)
    assert s > 0 and p < 0.01

def test_symmetry():
    r = np.random.default_rng(3); X, Y = r.normal(size=50), r.normal(1, size=60)
    assert np.isclose(mmd2_unbiased(X, Y, 1.0), mmd2_unbiased(Y, X, 1.0))

def test_changepoint_peak():
    r = np.random.default_rng(4); s = np.r_[r.normal(size=200), r.normal(2, size=200)]
    st = mmd_scan(s, 40); assert abs(np.nanargmax(st) - 200) <= 15
