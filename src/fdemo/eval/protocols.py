"""P1 seen attacks / P2 leave-one-attack-group-out / P3 external generators. Identity-disjoint folds."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fdemo import config
from fdemo.eval.metrics import threshold_at_bpcer
from fdemo.fusion.model import Scorer
from fdemo.fusion.policy import make_thresholds

OP_BPCER = 0.05   # operating point for per-attack APCER tables (chosen on dev, never on test)
RESULT_COLS = ["protocol", "split", "cfg", "vid", "label", "attack", "group", "actor",
               "score", "tau_op", "tau_review", "tau_deny"]


@dataclass
class Split:
    protocol: str
    name: str
    train: np.ndarray
    test: np.ndarray


def p1_splits(df) -> list:
    syn = df["group"] != "EXT"
    return [Split("P1", f"fold{f}", (syn & (df.fold != f)).to_numpy(), (syn & (df.fold == f)).to_numpy())
            for f in range(config.N_FOLDS)]


def p2_splits(df) -> list:
    syn, bona = df["group"] != "EXT", df["label"] == 0
    out = []
    for g in sorted(set(config.ATTACK_GROUP.values())):
        held = df["group"] == g
        for f in range(config.N_FOLDS):
            out.append(Split("P2", f"{g}/fold{f}", (syn & (df.fold != f) & ~held).to_numpy(),
                             (syn & (df.fold == f) & (bona | held)).to_numpy()))
    return out


def p3_splits(df) -> list:
    ext = df["group"] == "EXT"
    if not ext.any():
        return []
    syn = ~ext
    return [Split("P3", f"fold{f}", (syn & (df.fold != f)).to_numpy(),
                  ((syn & (df.fold == f) & (df.label == 0)) | ext).to_numpy())
            for f in range(config.N_FOLDS)]


def run_protocol(df, cfg, splits) -> pd.DataFrame:
    parts = []
    for sp in splits:
        tr, te = df[sp.train], df[sp.test]
        scorer = Scorer(cfg).fit(tr)
        dev = scorer.dev_scores(tr)
        y = tr["label"].to_numpy()
        th = make_thresholds(dev[y == 0], dev[y == 1])
        res = te[["vid", "label", "attack", "group", "actor"]].copy()
        res["score"] = scorer.predict(te)
        res["protocol"], res["split"], res["cfg"] = sp.protocol, sp.name, cfg
        res["tau_op"] = threshold_at_bpcer(dev[y == 0], OP_BPCER)
        res["tau_review"], res["tau_deny"] = th.review, th.deny
        parts.append(res)
    if not parts:
        return pd.DataFrame(columns=RESULT_COLS)
    return pd.concat(parts, ignore_index=True)[RESULT_COLS]
