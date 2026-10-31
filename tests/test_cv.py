import numpy as np
from kfp.cv import SpatioTemporalBlockCV
from kfp.metrics import summary, csi

def _data(n=600, r=np.random.default_rng(0)):
    return r.uniform(77, 81, n), r.uniform(28, 31, n), r.uniform(0, 365 * 5, n)

def test_blocks_disjoint_and_cover():
    lon, lat, t = _data(); cv = SpatioTemporalBlockCV(5, 1.0, 365)
    ids = cv.block_ids(lon, lat, t); seen = []
    for tr, te in cv.split(lon, lat, t):
        assert not set(ids[tr]) & set(ids[te]); seen += list(te)
    assert sorted(seen) == list(range(len(lon)))

def test_buffer_removes_neighbours():
    lon, lat, t = _data(); cv = SpatioTemporalBlockCV(4, 1.0, 365, time_buffer=10, space_buffer=0.2)
    for tr, te in cv.split(lon, lat, t):
        for i in tr[:100]:
            assert not np.any((np.abs(t[te] - t[i]) <= 10) & (np.abs(lon[te] - lon[i]) <= 0.2) & (np.abs(lat[te] - lat[i]) <= 0.2))

def test_metrics():
    y = np.array([0, 0, 1, 1]); p = np.array([.1, .4, .6, .9])
    s = summary(y, p); assert s["auc"] == 1.0 and csi(y, p) == 1.0
