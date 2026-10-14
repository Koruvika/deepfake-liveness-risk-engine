import numpy as np

from fdemo import config
from fdemo.io.video import probe, read_frames, sample_indices, sdk_capture, write_frames


def moving_square(n=30, h=360, w=640):
    frames = np.full((n, h, w, 3), 90, np.uint8)
    for t in range(n):
        frames[t, 100:160, 50 + 5 * t:110 + 5 * t] = 220
    return frames


def test_write_read_roundtrip(tmp_path):
    frames = moving_square()
    p = tmp_path / "a.mp4"
    write_frames(frames, p)
    back = read_frames(p)
    assert back.shape == frames.shape
    assert np.abs(back.astype(int) - frames.astype(int)).mean() < 6


def test_sdk_capture_normalises_size_fps_and_length(tmp_path):
    src = tmp_path / "src.mp4"
    write_frames(moving_square(100, 720, 1280), src, fps=30)
    dst = tmp_path / "dst.mp4"
    sdk_capture(src, dst)
    info = probe(dst)
    assert (info["width"], info["height"]) == (config.WIDTH, config.HEIGHT)
    assert abs(info["fps"] - config.FPS) < 0.01
    assert info["frames"] == config.N_FRAMES


def test_sdk_capture_preserves_aspect_ratio_with_padding(tmp_path):
    src = tmp_path / "src43.mp4"
    frames = np.full((100, 480, 640, 3), 200, np.uint8)
    write_frames(frames, src, fps=30)
    dst = tmp_path / "dst43.mp4"
    sdk_capture(src, dst)
    info = probe(dst)
    assert (info["width"], info["height"]) == (config.WIDTH, config.HEIGHT)
    assert info["frames"] == config.N_FRAMES
    back = read_frames(dst)
    assert back[:, :, 0].mean() < 20


def test_sample_indices():
    idx = sample_indices(80, 16)
    assert len(idx) == 16 and idx[0] == 0 and idx[-1] == 79
    assert np.all(np.diff(idx) > 0)
