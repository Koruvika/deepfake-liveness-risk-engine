"""Score one video with the final model and apply the policy (optionally with a FaceTec session JSON)."""
import dataclasses
import json
import subprocess
import tempfile
import time
from pathlib import Path

import joblib
import pandas as pd

from fdemo import config
from fdemo.features import extract_video
from fdemo.fusion.failures import signal_failure_mask
from fdemo.fusion.policy import decide
from fdemo.fusion.vendor_adapter import VendorPayloadError, parse_vendor
from fdemo.io.video import sdk_capture
from fdemo.signals import SIGNAL_PREFIX


def contributions(scorer, X, medians, risk) -> list:
    """Model-agnostic: risk drop when a signal group is replaced by its training median."""
    out = []
    for name, prefix in SIGNAL_PREFIX.items():
        cols = [c for c in X.columns if c.startswith(prefix)]
        if not cols:
            continue
        Xm = X.copy()
        for c in cols:
            Xm[c] = medians.get(c)
        out.append((name, float(risk - scorer.predict(Xm)[0])))
    return sorted(out, key=lambda kv: kv[1], reverse=True)[:3]


def scan_video(video, vendor_json=None, photo_id=False, expected_ref=None) -> dict:
    if not Path(config.BUNDLE).exists():
        raise SystemExit(f"model bundle not found at {config.BUNDLE}; run `fdemo bench` first")
    bundle = joblib.load(config.BUNDLE)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        cap = Path(td) / "capture.mp4"
        try:
            sdk_capture(video, cap)
        except subprocess.CalledProcessError as e:
            raise SystemExit(f"could not decode video: {e}") from e
        row = extract_video(f"scan_{Path(video).stem}", cap, cache_dir=Path(td))
    X = pd.DataFrame([row]).reindex(columns=bundle["columns"])
    scorer, th = bundle["scorer"], bundle["thresholds"]
    risk = float(scorer.predict(X)[0])
    vendor = None
    if vendor_json:
        try:
            vendor = parse_vendor(json.loads(Path(vendor_json).read_text()), expected_ref=expected_ref)
        except (json.JSONDecodeError, VendorPayloadError) as e:
            raise SystemExit(f"invalid vendor session JSON: {e}") from e
    nan_free_cols = bundle.get("nan_free_cols", [])
    err_keys = [k for k in row if k.startswith("err_")]
    # Reindex so a nan_free_cols feature that's entirely absent from `row` (not merely NaN) still reads as
    # NaN here, the same as it would in X above.
    feat_row = pd.DataFrame([row]).reindex(columns=nan_free_cols + err_keys)
    signal_failure = bool(signal_failure_mask(feat_row, nan_free_cols)[0])
    d = decide(risk, th, vendor=vendor, photo_id=photo_id, no_face=bool(row.get("no_face", False)),
               signal_failure=signal_failure)
    return {
        "video": str(video), "risk": risk, "decision": d.action, "reason": d.reason,
        "thresholds": {"review": th.review, "deny": th.deny},
        "top_signals": contributions(scorer, X, bundle["medians"], risk),
        "vendor": dataclasses.asdict(vendor) if vendor else None,
        "latency_s": time.perf_counter() - t0,
    }
