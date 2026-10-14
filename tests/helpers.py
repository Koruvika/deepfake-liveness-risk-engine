import cv2
import numpy as np


def circle_points(cx=320.0, cy=180.0, r=90.0, n=36):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1).astype(np.float32)


def noisy_frame(sigma, seed, shape=(360, 640)):
    g = np.random.default_rng(seed).normal(128, sigma, shape).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def smooth_frames(T=20, h=360, w=640, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.indices((h, w))
    base = (100 + 50 * np.sin(x / 80.0) + 30 * np.cos(y / 60.0)).astype(np.float32)
    return np.stack([
        np.clip(base + rng.normal(0, 3, (h, w)), 0, 255).astype(np.uint8)[..., None].repeat(3, 2)
        for _ in range(T)
    ])


def fake_lms(T, seed=0, box=((220, 80), (420, 280))):
    """478 pseudo-landmarks inside a face box, identical in every frame."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(box[0], box[1], (478, 2)).astype(np.float32)
    return np.repeat(pts[None], T, axis=0)
