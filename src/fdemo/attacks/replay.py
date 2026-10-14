"""A3: simulated screen recapture (resampling, gamma/white balance, moiré, glare, perspective, new sensor noise)."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ReplayParams:
    scale: float
    gamma: float
    wb: np.ndarray
    f1: float
    f2: float
    th1: float
    th2: float
    amp: float
    drift: float
    gx: float
    gy: float
    gr: float
    gs: float
    H: np.ndarray
    noise: float


def sample_params(rng, h, w) -> ReplayParams:
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jit = rng.uniform(0.02, 0.05) * np.float32([w, h])
    dst = (src + rng.uniform(-1, 1, (4, 2)).astype(np.float32) * jit).astype(np.float32)
    th1 = rng.uniform(0, np.pi)
    return ReplayParams(
        scale=rng.uniform(0.35, 0.6), gamma=rng.uniform(0.75, 1.3), wb=rng.uniform(0.9, 1.1, 3).astype(np.float32),
        f1=rng.uniform(0.15, 0.35), f2=rng.uniform(0.15, 0.35), th1=th1, th2=th1 + rng.uniform(0.05, 0.3),
        amp=rng.uniform(3, 8), drift=rng.uniform(0.05, 0.2), gx=rng.uniform(0, w), gy=rng.uniform(0, h),
        gr=rng.uniform(0.15, 0.4) * w, gs=rng.uniform(10, 40), H=cv2.getPerspectiveTransform(src, dst),
        noise=rng.uniform(2, 4),
    )


def moire(h, w, p, t) -> np.ndarray:
    y, x = np.indices((h, w), dtype=np.float32)
    g1 = np.sin(2 * np.pi * p.f1 * (x * np.cos(p.th1) + y * np.sin(p.th1)) + p.drift * t)
    g2 = np.sin(2 * np.pi * p.f2 * (x * np.cos(p.th2) + y * np.sin(p.th2)))
    return (p.amp * g1 * g2).astype(np.float32)


def glare(h, w, p) -> np.ndarray:
    y, x = np.indices((h, w), dtype=np.float32)
    return (p.gs * np.exp(-((x - p.gx) ** 2 + (y - p.gy) ** 2) / (2 * p.gr ** 2))).astype(np.float32)


def replay_frame(frame, p, t, rng, gl) -> np.ndarray:
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (int(w * p.scale), int(h * p.scale)), interpolation=cv2.INTER_AREA)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    img = 255.0 * np.power(np.clip(img * p.wb / 255.0, 0, 1), p.gamma)
    img += (moire(h, w, p, t) + gl)[..., None]
    img = cv2.warpPerspective(img, p.H, (w, h), borderMode=cv2.BORDER_REFLECT)
    img += rng.normal(0, p.noise, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def generate(target_frames, rng, source_frame=None) -> np.ndarray:
    h, w = target_frames.shape[1:3]
    p = sample_params(rng, h, w)
    gl = glare(h, w, p)
    return np.stack([replay_frame(f, p, t, rng, gl) for t, f in enumerate(target_frames)])
