import numpy as np
import pandas as pd

from fdemo import config
from fdemo.data.prepare import make_splits


def synthetic_dataset(per_actor=5, seed=0, with_ext=False) -> pd.DataFrame:
    """Manifest+features-shaped frame: attacks shift a few features; one feature is pure noise."""
    rng = np.random.default_rng(seed)
    folds = make_splits(config.ACTORS, config.N_FOLDS, config.SEED)
    rows = []

    def feats(shift):
        return {
            "pad_real_mean": float(np.clip(rng.normal(0.8 - 0.3 * shift, 0.15), 0, 1)),
            "pad_real_min": float(rng.uniform(0, 1)), "pad_real_low4": float(rng.uniform(0, 1)),
            "pad_real_std": float(rng.uniform(0, 0.2)),
            "sbi_mean": float(np.clip(rng.normal(0.3 + 0.2 * shift, 0.15), 0, 1)),
            "sbi_top4": float(rng.uniform(0, 1)), "sbi_std": float(rng.uniform(0, 0.2)),
            "bn_core_bg": float(rng.normal(shift, 0.5)), "sp_peaks": float(rng.normal(0.5 * shift, 0.5)),
            "tm_dup_rate": float(rng.normal(0.3 * shift, 0.2)), "id_drift_std": float(rng.normal(0.2 * shift, 0.2)),
            "rp_snr": float(rng.normal(-shift, 1.0)), "noise_col": float(rng.normal()),
        }

    for a in config.ACTORS:
        for k in range(per_actor):
            rows.append({"vid": f"bona_{a}_{k}", "label": 0, "attack": "bona", "group": "bona", "actor": a,
                         "fold": folds[a], **feats(0.0)})
            for atk in config.ATTACKS:
                rows.append({"vid": f"{atk}_{a}_{k}", "label": 1, "attack": atk, "group": config.ATTACK_GROUP[atk],
                             "actor": a, "fold": folds[a], **feats(1.5)})
    if with_ext:
        for i in range(10):
            rows.append({"vid": f"ext_{i}", "label": 1, "attack": "EXT_Hallo", "group": "EXT", "actor": -1,
                         "fold": -1, **feats(1.0)})
    return pd.DataFrame(rows)
