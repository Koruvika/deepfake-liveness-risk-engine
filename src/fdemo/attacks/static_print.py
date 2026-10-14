"""A5: a single frame 'printed' (desaturated, lower contrast, paper texture) and held by hand (small affine drift)."""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


def paper_texture(shape, rng) -> np.ndarray:
    n = cv2.GaussianBlur(rng.normal(0, 1, shape[:2]).astype(np.float32), (0, 0), 1.2)
    return n * rng.uniform(3, 6)


def print_look(frame, rng) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= rng.uniform(0.65, 0.85)
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    img = (img - 128) * rng.uniform(0.85, 0.95) + 128
    return img + paper_texture(frame.shape, rng)[..., None]


def walk(n, rng, step) -> np.ndarray:
    return gaussian_filter1d(np.cumsum(rng.normal(0, step, n)), 3)


def generate(target_frames, rng, source_frame=None) -> np.ndarray:
    T, h, w = target_frames.shape[:3]
    base = print_look(target_frames[T // 2], rng)
    rot, tx, ty = walk(T, rng, 0.15), walk(T, rng, 0.6), walk(T, rng, 0.6)
    sc = 1 + walk(T, rng, 0.002)
    out = []
    for t in range(T):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), float(rot[t]), float(sc[t]))
        M[:, 2] += (tx[t], ty[t])
        img = cv2.warpAffine(base, M, (w, h), borderMode=cv2.BORDER_REFLECT) + rng.normal(0, 2, base.shape)
        out.append(np.clip(img, 0, 255).astype(np.uint8))
    return np.stack(out)
