"""Per-video context shared by all signals: frames, landmarks, sampled indices, cached detections."""
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fdemo import config
from fdemo.io.video import read_frames, sample_indices


@dataclass
class VideoCtx:
    vid: str
    frames: np.ndarray        # (T, H, W, 3) BGR uint8
    lms: np.ndarray           # (T, 478, 2) float32, NaN where no face
    face_ok: np.ndarray       # (T,) bool
    sampled: np.ndarray       # (N_SAMPLED,) frame indices for per-frame signals
    fps: float = float(config.FPS)
    extras: dict = field(default_factory=dict)

    @property
    def no_face(self) -> bool:
        return bool(np.mean(self.face_ok) < 0.5)


def compute_landmarks(frames, landmarker):
    T = len(frames)
    lms = np.full((T, 478, 2), np.nan, np.float32)
    ok = np.zeros(T, bool)
    for t, f in enumerate(frames):
        p = landmarker(f)
        if p is not None:
            lms[t], ok[t] = p, True
    return lms, ok


def build_ctx(path, vid, cache_dir=None) -> VideoCtx:
    from fdemo import models

    frames = read_frames(path, config.N_FRAMES)
    cache = Path(cache_dir or config.CACHE / "ctx") / f"{vid}.npz"
    st = Path(path).stat()
    lms = ok = None
    if cache.exists():
        try:
            with np.load(cache) as d:
                if (len(d["lms"]) == len(frames) and int(d["src_mtime_ns"]) == st.st_mtime_ns
                        and int(d["src_size"]) == st.st_size):
                    lms, ok = d["lms"], d["ok"]
        except (OSError, ValueError, zipfile.BadZipFile, KeyError):
            lms = ok = None
    if lms is None:
        lms, ok = compute_landmarks(frames, models.landmarker())
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(cache.stem + ".tmp.npz")
        np.savez_compressed(tmp, lms=lms, ok=ok, src_mtime_ns=st.st_mtime_ns, src_size=st.st_size)
        os.replace(tmp, cache)
    return VideoCtx(vid, frames, lms, ok, sample_indices(len(frames), config.N_SAMPLED))


def sampled_detections(ctx) -> list:
    """RetinaFace detections on the sampled frames, computed once per ctx."""
    if "dets" not in ctx.extras:
        from fdemo import models

        det = models.detector()
        ctx.extras["dets"] = [det(ctx.frames[i]) for i in ctx.sampled]
    return ctx.extras["dets"]
