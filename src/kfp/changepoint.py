"""Kernel changepoint scan: sliding two-window MMD^2 statistic (basic)."""
import numpy as np
from .mmd import mmd2_unbiased, median_heuristic


def mmd_scan(series, window: int, bandwidth=None):
    s = np.asarray(series, float)
    s = s[:, None] if s.ndim == 1 else s
    bw = bandwidth or median_heuristic(s)
    stats = np.full(len(s), np.nan)
    for t in range(window, len(s) - window + 1):
        stats[t] = mmd2_unbiased(s[t - window:t], s[t:t + window], bw)
    return stats
