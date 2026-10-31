"""Verification metrics: AUC, PR-AUC, Brier, CSI."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def auc(y, p): return float(roc_auc_score(y, p))
def pr_auc(y, p): return float(average_precision_score(y, p))
def brier(y, p): return float(brier_score_loss(y, p))


def csi(y, p, threshold=0.5):
    """Critical success index = hits / (hits + misses + false alarms)."""
    y = np.asarray(y).astype(bool); f = np.asarray(p) >= threshold
    h, m, fa = np.sum(y & f), np.sum(y & ~f), np.sum(~y & f)
    return float(h / (h + m + fa)) if (h + m + fa) else np.nan


def summary(y, p, threshold=0.5):
    return {"auc": auc(y, p), "pr_auc": pr_auc(y, p), "brier": brier(y, p), "csi": csi(y, p, threshold)}
