"""Identity drift: face swaps flicker between source and target identity across frames."""
import numpy as np

from fdemo import models
from fdemo.ctx import sampled_detections
from fdemo.signals.base import nan_features

KEYS = ["id_drift_std", "id_drift_min"]


def score(ctx) -> dict:
    arc = models.arcface()
    dets = sampled_detections(ctx)
    embs = []
    for i, d in zip(ctx.sampled, dets):
        if d is None:
            continue
        try:
            embs.append(arc.embed(ctx.frames[i], d.kps))
        except ValueError:
            continue
    if len(embs) < 3:
        return nan_features(KEYS)
    E = np.stack(embs)
    m = E.mean(0)
    sims = E @ (m / np.linalg.norm(m))
    return {"id_drift_std": float(sims.std()), "id_drift_min": float(sims.min())}
