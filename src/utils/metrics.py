import numpy as np


def smape(y_true, y_pred, eps=1e-7):
    num = np.abs(y_pred - y_true)
    den = np.abs(y_true) + np.abs(y_pred) + eps
    return 200.0 * np.mean(num / den)


def kl_div(y_true, y_pred, eps=1e-7):
    p = np.clip(y_true, eps, 1.0)
    q = np.clip(y_pred, eps, 1.0)
    return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=1)))


def bias(y_true, y_pred):
    """Mean bias per item."""
    return np.mean(y_pred - y_true, axis=0)
