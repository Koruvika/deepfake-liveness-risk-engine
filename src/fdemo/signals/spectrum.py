"""Azimuthal power spectrum of the face crop: high-frequency balance, slope and moiré/upsampling peaks."""
import cv2
import numpy as np

from fdemo.signals.base import nan_features

N_BINS = 32
KEYS = ["sp_hf_ratio", "sp_slope", "sp_peaks", "sp_peaks_max"]


def face_crop_gray(frame, lms, size=128):
    x0, y0 = np.nanmin(lms, 0)
    x1, y1 = np.nanmax(lms, 0)
    cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 0.55
    H, W = frame.shape[:2]
    xa, ya = int(max(0, cx - half)), int(max(0, cy - half))
    xb, yb = int(min(W, cx + half)), int(min(H, cy + half))
    crop = frame[ya:yb, xa:xb]
    if crop.size == 0:
        return None
    ch, cw = crop.shape[:2]
    if not (0.8 <= cw / ch <= 1.25):
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)


def log_spectrum(gray):
    h, w = gray.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2((gray - gray.mean()) * win))))


def radial_index(h, w, nbins=N_BINS):
    y, x = np.indices((h, w))
    r = np.hypot(x - w / 2, y - h / 2) / (min(h, w) / 2)
    return np.minimum((r * nbins).astype(int), nbins - 1), r


def spectrum_features(gray) -> dict:
    gray = np.asarray(gray, np.float32)
    mag = log_spectrum(gray)
    idx, r = radial_index(*gray.shape)
    inner = (r < 1.0).ravel()
    counts = np.maximum(np.bincount(idx.ravel()[inner], minlength=N_BINS), 1)
    prof = np.bincount(idx.ravel()[inner], mag.ravel()[inner], N_BINS) / counts
    hf = prof[N_BINS // 2:].mean() / (prof[: N_BINS // 2].mean() + 1e-6)
    centers = (np.arange(N_BINS) + 0.5) / N_BINS
    slope = np.polyfit(np.log(centers[4:]), prof[4:], 1)[0]
    resid = mag - prof[idx]
    ring = (r > 0.25) & (r < 1.0)
    z = (resid[ring] - resid[ring].mean()) / (resid[ring].std() + 1e-6)
    # z > 3.5 (not 5): moiré is partly smoothed by the final H.264 encode; 3.5 was the highest threshold that
    # still fired on simulated replays while staying at 0 on bona fide clips during plan verification
    return {"sp_hf_ratio": float(hf), "sp_slope": float(slope), "sp_peaks": float((z > 3.5).mean() * 1000)}


def score(ctx) -> dict:
    feats = []
    for i in ctx.sampled:
        if not ctx.face_ok[i]:
            continue
        crop = face_crop_gray(ctx.frames[i], ctx.lms[i])
        if crop is not None:
            feats.append(spectrum_features(crop))
    if not feats:
        return nan_features(KEYS)
    peaks = np.array([f["sp_peaks"] for f in feats])
    return {
        "sp_hf_ratio": float(np.mean([f["sp_hf_ratio"] for f in feats])),
        "sp_slope": float(np.mean([f["sp_slope"] for f in feats])),
        "sp_peaks": float(peaks.mean()),
        "sp_peaks_max": float(peaks.max()),
    }
