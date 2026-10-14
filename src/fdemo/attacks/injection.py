"""A4: virtual-camera injection of a face-swap stream (OBS-like): 15 fps generator up-sampled to 25 fps,
denoise or temporal blending, then a lossy re-encode before the SDK capture."""
import tempfile
from pathlib import Path

import cv2
import numpy as np

from fdemo.io.video import read_frames, write_frames


def resample_fps(frames, src_fps=15, dst_fps=25) -> np.ndarray:
    n = len(frames)
    k = np.floor(np.arange(n) * src_fps / dst_fps)            # 15-fps generator tick
    idx = np.minimum(np.round(k * dst_fps / src_fps), n - 1)  # real-time source frame of that tick
    return frames[idx.astype(int)]


def temporal_smooth(frames, alpha) -> np.ndarray:
    out = frames.astype(np.float32).copy()
    for t in range(1, len(out)):
        out[t] = alpha * out[t] + (1 - alpha) * out[t - 1]
    return np.clip(out, 0, 255).astype(np.uint8)


def denoise(frames, h) -> np.ndarray:
    return np.stack([cv2.fastNlMeansDenoisingColored(f, None, h, h, 5, 11) for f in frames])


def reencode(frames, crf) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        p = write_frames(frames, Path(td) / "vcam.mp4", crf=crf)
        return read_frames(p, len(frames))


def generate(a2_frames, rng) -> np.ndarray:
    fr = resample_fps(a2_frames)
    if rng.random() < 0.5:
        fr = denoise(fr, float(rng.uniform(4, 8)))
    else:
        fr = temporal_smooth(fr, float(rng.uniform(0.55, 0.75)))
    return reencode(fr, int(rng.integers(28, 34)))
