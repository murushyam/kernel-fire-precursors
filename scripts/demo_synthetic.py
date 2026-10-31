"""Synthetic end-to-end demo (no real data; illustrates the API only)."""
import numpy as np
from kfp.mmd import mmd_permutation_test
from kfp.embeddings import DistributionRegressor
from kfp.cv import SpatioTemporalBlockCV
from kfp.metrics import summary
from kfp.bandit import gp_ucb

r = np.random.default_rng(42)
# 1. precursor MMD: pre-fire (drier, warmer) vs baseline samples of (soil moisture, VPD)
base = np.c_[r.normal(0.30, 0.05, 200), r.normal(1.0, 0.3, 200)]
pre = np.c_[r.normal(0.25, 0.05, 200), r.normal(1.3, 0.3, 200)]
print("MMD^2, p =", mmd_permutation_test(pre, base, n_perm=300, rng=0))

# 2. distribution regression on sub-grid bags -> fire probability, with blocked CV
n = 240; lon, lat, t = r.uniform(77.5, 81, n), r.uniform(28.7, 31.5, n), r.uniform(0, 1825, n)
dry = r.uniform(0, 1, n)
bags = [np.c_[r.normal(0.3 - 0.1 * d, 0.05, 25), r.normal(1 + d, 0.3, 25)] for d in dry]
y = (r.uniform(size=n) < dry).astype(int)
cv = SpatioTemporalBlockCV(5, 1.0, 365, time_buffer=8, space_buffer=0.25)

def cv_auc(params):
    a = 10 ** params[0]; ps = np.zeros(n); ok = np.zeros(n, bool)
    for tr, te in cv.split(lon, lat, t):
        m = DistributionRegressor(alpha=a).fit([bags[i] for i in tr], y[tr])
        ps[te] = np.clip(m.predict([bags[i] for i in te]), 0, 1); ok[te] = True
    return summary(y[ok], ps[ok])["auc"]

best, hist = gp_ucb(cv_auc, np.linspace(-5, 0, 11)[:, None], n_iter=6)
print("GP-UCB best log10(alpha) =", best[0], "AUC =", max(h[1] for h in hist))
