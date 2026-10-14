"""Remote photoplethysmography with POS (Wang et al. 2017) on forehead and cheeks."""
import numpy as np
from scipy.signal import butter, detrend, filtfilt

from fdemo.face.geometry import ROIS, roi_mask
from fdemo.signals.base import nan_features

P = np.array([[0, 1, -1], [-2, 1, 1]], float)
KEYS = ["rp_snr", "rp_snr_max", "rp_agree"]


def pos(rgb, fps) -> np.ndarray:
    T = len(rgb)
    l = int(round(1.6 * fps))
    H = np.zeros(T)
    for n in range(T - l + 1):
        C = rgb[n:n + l].T
        Cn = C / (C.mean(axis=1, keepdims=True) + 1e-9)
        S = P @ Cn
        h = S[0] + (S[0].std() / (S[1].std() + 1e-9)) * S[1]
        H[n:n + l] += h - h.mean()
    return H


def band_snr(sig, fps, lo=0.7, hi=3.0, nfft=2048):
    sig = detrend(sig)
    b, a = butter(3, [lo / (fps / 2), hi / (fps / 2)], btype="band")
    sig = filtfilt(b, a, sig)
    if sig.std() < 1e-9:
        return float("nan"), float("nan")
    f = np.fft.rfftfreq(nfft, 1 / fps)
    pw = np.abs(np.fft.rfft(sig * np.hanning(len(sig)), nfft)) ** 2
    band = (f >= lo) & (f <= hi)
    fb, pb = f[band], pw[band]
    if pb.sum() == 0:
        return float("nan"), float("nan")
    f0 = fb[pb.argmax()]
    m = (np.abs(fb - f0) <= 0.15) | (np.abs(fb - 2 * f0) <= 0.15)
    return float(10 * np.log10(pb[m].sum() / (pb[~m].sum() + 1e-12) + 1e-12)), float(f0)


def roi_traces(frames, lms, ok):
    T = len(frames)
    traces = {name: np.full((T, 3), np.nan) for name in ROIS}
    for t in range(T):
        if not ok[t]:
            continue
        for name, idx in ROIS.items():
            m = roi_mask(lms[t], idx, frames[t].shape) > 0
            if m.sum() >= 20:
                traces[name][t] = frames[t][m].mean(axis=0)[::-1]  # BGR -> RGB
    out = {}
    for name, tr in traces.items():
        valid = ~np.isnan(tr[:, 0])
        if valid.sum() < 0.8 * T:
            return None
        x = np.arange(T)
        out[name] = np.stack([np.interp(x, x[valid], tr[valid, c]) for c in range(3)], 1)
    return out


def score(ctx) -> dict:
    traces = roi_traces(ctx.frames, ctx.lms, ctx.face_ok)
    if traces is None:
        return nan_features(KEYS)
    res = {name: band_snr(pos(tr, ctx.fps), ctx.fps) for name, tr in traces.items()}
    snrs = [s for s, _ in res.values()]
    return {
        "rp_snr": float(np.mean(snrs)),
        "rp_snr_max": float(np.max(snrs)),
        "rp_agree": float(abs(res["left_cheek"][1] - res["right_cheek"][1])),
    }
