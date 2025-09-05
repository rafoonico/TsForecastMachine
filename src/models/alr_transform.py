import numpy as np


def alr_transform(shares: np.ndarray, ref_idx: int = -1, eps: float = 1e-6):
    s = np.clip(shares, eps, 1.0)
    s = s / s.sum(axis=1, keepdims=True)
    ref = s[:, [ref_idx]]
    others = np.delete(s, ref_idx, axis=1)
    z = np.log(others / ref)
    return z


def alr_inverse(z: np.ndarray, ref_idx: int = -1):
    n, k_1 = z.shape
    logits = np.zeros((n, k_1 + 1))
    logits[:, :k_1] = z
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)
