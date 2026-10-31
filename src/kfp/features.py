"""Precursor windows: samples of predictors in a window before events vs matched baselines."""
from __future__ import annotations
import numpy as np


def precursor_window(series: np.ndarray, event_idx, lead: int, width: int):
    """Stack samples from [t-lead-width, t-lead) for each event time t.

    series: (T, d) array. Returns (n_samples, d)."""
    series = np.asarray(series)
    if series.ndim == 1:
        series = series[:, None]
    out = [series[t - lead - width:t - lead] for t in event_idx if t - lead - width >= 0]
    return np.concatenate(out) if out else np.empty((0, series.shape[1]))


def baseline_indices(n_time: int, event_idx, exclusion: int, n: int, rng=None):
    """Random non-event times at least `exclusion` steps from any event."""
    rng = np.random.default_rng(rng)
    ev = np.asarray(event_idx)
    cand = [t for t in range(n_time) if ev.size == 0 or np.min(np.abs(ev - t)) > exclusion]
    return rng.choice(cand, size=min(n, len(cand)), replace=False)
