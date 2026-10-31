"""GP-UCB search over a discrete candidate grid (basic)."""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel


def gp_ucb(objective, candidates, n_iter=15, n_init=3, beta=2.0, rng=0):
    """Maximise objective(c) over rows of `candidates` (n, d). Returns (best, history)."""
    C = np.atleast_2d(np.asarray(candidates, float))
    if C.shape[0] == 1 and len(candidates) > 1: C = C.T
    r = np.random.default_rng(rng)
    idx = list(r.choice(len(C), size=min(n_init, len(C)), replace=False)); ys = [objective(C[i]) for i in idx]
    gp = GaussianProcessRegressor(Matern(nu=2.5) + WhiteKernel(1e-3), normalize_y=True)
    for _ in range(max(0, n_iter - len(idx))):
        if len(idx) == len(C): break
        gp.fit(C[idx], ys)
        mu, sd = gp.predict(C, return_std=True)
        ucb = mu + np.sqrt(beta) * sd; ucb[idx] = -np.inf
        j = int(np.argmax(ucb)); idx.append(j); ys.append(objective(C[j]))
    b = int(np.argmax(ys))
    return C[idx[b]], list(zip(idx, ys))
