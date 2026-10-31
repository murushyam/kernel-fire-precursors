"""Spatiotemporal blocked cross-validation with buffer (leakage control)."""
from __future__ import annotations
import numpy as np


class SpatioTemporalBlockCV:
    """Assign samples to (space block x time block) groups and hold out whole blocks.

    Training samples within `time_buffer` (time units) or `space_buffer` (deg)
    of any test sample are dropped from the training set."""

    def __init__(self, n_splits=5, space_block=1.0, time_block=365, time_buffer=0.0,
                 space_buffer=0.0, random_state=0):
        self.n_splits, self.space_block, self.time_block = n_splits, space_block, time_block
        self.time_buffer, self.space_buffer, self.random_state = time_buffer, space_buffer, random_state

    def block_ids(self, lon, lat, t):
        bx = np.floor(np.asarray(lon) / self.space_block).astype(int)
        by = np.floor(np.asarray(lat) / self.space_block).astype(int)
        bt = np.floor(np.asarray(t) / self.time_block).astype(int)
        _, ids = np.unique(np.stack([bx, by, bt], 1), axis=0, return_inverse=True)
        return ids.ravel()

    def split(self, lon, lat, t):
        lon, lat, t = map(lambda a: np.asarray(a, float), (lon, lat, t))
        ids = self.block_ids(lon, lat, t)
        ub = np.unique(ids)
        if len(ub) < self.n_splits:
            raise ValueError(f"only {len(ub)} blocks for {self.n_splits} splits")
        rng = np.random.default_rng(self.random_state)
        folds = np.array_split(rng.permutation(ub), self.n_splits)
        for fb in folds:
            test = np.isin(ids, fb)
            train = ~test
            if self.time_buffer > 0 or self.space_buffer > 0:
                te = np.where(test)[0]
                for i in np.where(train)[0]:
                    near_t = np.abs(t[te] - t[i]) <= self.time_buffer
                    near_s = (np.abs(lon[te] - lon[i]) <= self.space_buffer) & (np.abs(lat[te] - lat[i]) <= self.space_buffer)
                    if np.any(near_t & near_s):
                        train[i] = False
            yield np.where(train)[0], np.where(test)[0]
