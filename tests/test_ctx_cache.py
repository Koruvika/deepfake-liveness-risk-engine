"""Landmark cache correctness in fdemo.ctx.build_ctx: hit/miss on file identity, atomic writes,
and graceful recovery from a corrupt cache file."""
import time

import numpy as np

from fdemo import ctx as ctx_mod
from fdemo.io.video import write_frames


class FakeLandmarker:
    def __init__(self):
        self.calls = 0

    def __call__(self, frame):
        self.calls += 1
        return np.ones((478, 2), np.float32)


def make_video(tmp_path, name="v.mp4", seed=0, n=5):
    frames = np.random.default_rng(seed).integers(0, 255, (n, 32, 32, 3), dtype=np.uint8)
    return write_frames(frames, tmp_path / name)


def test_cache_hit_skips_landmarker(tmp_path, monkeypatch):
    fake = FakeLandmarker()
    monkeypatch.setattr("fdemo.models.landmarker", lambda: fake)
    video = make_video(tmp_path)
    cache_dir = tmp_path / "cache"

    ctx_mod.build_ctx(video, "v", cache_dir=cache_dir)
    first_calls = fake.calls
    assert first_calls > 0

    ctx_mod.build_ctx(video, "v", cache_dir=cache_dir)
    assert fake.calls == first_calls


def test_cache_invalidated_when_source_file_changes(tmp_path, monkeypatch):
    fake = FakeLandmarker()
    monkeypatch.setattr("fdemo.models.landmarker", lambda: fake)
    video = make_video(tmp_path, seed=0)
    cache_dir = tmp_path / "cache"

    ctx_mod.build_ctx(video, "v", cache_dir=cache_dir)
    first_calls = fake.calls

    time.sleep(0.01)
    make_video(tmp_path, seed=1)  # rewrite same path with different content/size/mtime
    ctx_mod.build_ctx(video, "v", cache_dir=cache_dir)
    assert fake.calls > first_calls


def test_corrupt_cache_file_is_recomputed_without_raising(tmp_path, monkeypatch):
    fake = FakeLandmarker()
    monkeypatch.setattr("fdemo.models.landmarker", lambda: fake)
    video = make_video(tmp_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "v.npz").write_bytes(b"garbage")

    out = ctx_mod.build_ctx(video, "v", cache_dir=cache_dir)
    assert fake.calls > 0
    assert out.lms.shape[0] == out.frames.shape[0]
