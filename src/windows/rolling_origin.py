import numpy as np


def rolling_splits(n, train_min, val, test, step):
    """Generate rolling-origin time series splits."""
    i = train_min
    while i + val + test <= n:
        yield (slice(0, i), slice(i, i + val), slice(i + val, i + val + test))
        i += step
