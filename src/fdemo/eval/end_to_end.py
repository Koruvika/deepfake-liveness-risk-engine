"""How the whole system responds: vendor-only vs vendor + risk engine, under three vendor scenarios.

V-standin        : MiniFASNet (p_real >= 0.5) plays FaceTec's livenessProven.
V-bypassed       : the vendor returns livenessProven=true for every session (injection got past it).
V-ideal-bonafide : the vendor never rejects a genuine user; attacks keep the stand-in's verdict."""
import numpy as np
import pandas as pd

from fdemo.fusion.failures import signal_failure_mask
from fdemo.fusion.policy import Thresholds, decide
from fdemo.fusion.vendor_adapter import VendorResult

SCENARIOS = ["V-standin", "V-bypassed", "V-ideal-bonafide"]
STANDIN_THRESHOLD = 0.5


def vendor_proven(scenario, label, pad_real_mean) -> np.ndarray:
    standin = np.nan_to_num(np.asarray(pad_real_mean, float), nan=0.0) >= STANDIN_THRESHOLD
    if scenario == "V-standin":
        return standin
    if scenario == "V-bypassed":
        return np.ones_like(standin, bool)
    if scenario == "V-ideal-bonafide":
        return np.where(np.asarray(label) == 0, True, standin)
    raise ValueError(scenario)


def _decisions(df, proven, signal_failure):
    acts, reasons = [], []
    for p, r, tr, td, sf in zip(proven, df.score, df.tau_review, df.tau_deny, signal_failure):
        d = decide(float(r), Thresholds(float(tr), float(td)), vendor=VendorResult(bool(p), None, None, None, "sim"),
                   signal_failure=bool(sf))
        acts.append(d.action)
        reasons.append(d.reason)
    return np.array(acts), np.array(reasons)


def _merge(res, features):
    return res.reset_index(drop=True).merge(features, on="vid", how="left")


def e2e_table(res, features, nan_free_cols=None) -> pd.DataFrame:
    """`features` must carry `vid`, `pad_real_mean`, any `err_*` columns, and the feature columns named in
    `nan_free_cols` (the same rule `fdemo scan` applies, so the two can't disagree about what counts as a
    signal failure)."""
    df = _merge(res, features)
    sf = signal_failure_mask(df, nan_free_cols or [])
    rows = []
    for sc in SCENARIOS:
        proven = vendor_proven(sc, df.label.to_numpy(), df.pad_real_mean.to_numpy())
        modes = {"vendor-only": np.where(proven, "allow", "deny"), "vendor+engine": _decisions(df, proven, sf)[0]}
        for mode, acts in modes.items():
            for attack in df.attack.unique():
                mask = (df.attack == attack).to_numpy()
                a = acts[mask]
                n_videos = int(df.loc[mask, "vid"].nunique())
                rows.append({"scenario": sc, "mode": mode, "attack": attack, "n": len(a), "n_videos": n_videos,
                             "allow": float((a == "allow").mean()), "review": float((a == "review").mean()),
                             "deny": float((a == "deny").mean())})
    return pd.DataFrame(rows)


def bpcer_breakdown(res, features, nan_free_cols=None) -> pd.DataFrame:
    """See `e2e_table` for the `features`/`nan_free_cols` contract."""
    df = _merge(res, features)
    sf = signal_failure_mask(df, nan_free_cols or [])
    bona = (df.label == 0).to_numpy()
    rows = []
    for sc in SCENARIOS:
        proven = vendor_proven(sc, df.label.to_numpy(), df.pad_real_mean.to_numpy())
        acts, reasons = _decisions(df, proven, sf)
        rows.append({"scenario": sc,
                     "deny_by_vendor": float((reasons[bona] == "vendor_liveness_not_proven").mean()),
                     "deny_by_engine": float((reasons[bona] == "risk_high").mean()),
                     "review_signal_failure": float((reasons[bona] == "signal_failure").mean()),
                     "review": float((acts[bona] == "review").mean()),
                     "allow": float((acts[bona] == "allow").mean())})
    return pd.DataFrame(rows)
