import numpy as np


def topk_mean(values, k) -> float:
    v = np.asarray(values, float)
    v = np.sort(v[np.isfinite(v)])[::-1]
    if len(v) == 0:
        return float("nan")
    return float(v[: max(1, min(k, len(v)))].mean())


def nan_features(keys) -> dict:
    return {k: float("nan") for k in keys}
