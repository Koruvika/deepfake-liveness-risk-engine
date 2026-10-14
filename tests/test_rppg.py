from types import SimpleNamespace

import numpy as np

from fdemo.signals import rppg
from helpers import fake_lms


def test_pos_recovers_pulse_frequency():
    fps, t = 25, np.arange(80) / 25
    rng = np.random.default_rng(0)
    pulse = np.sin(2 * np.pi * 1.2 * t)
    rgb = np.stack([120 + 0.3 * pulse, 100 + 0.8 * pulse, 90 + 0.5 * pulse], 1) + rng.normal(0, 0.1, (80, 3))
    snr, f0 = rppg.band_snr(rppg.pos(rgb, fps), fps)
    assert abs(f0 - 1.2) <= 0.2
    noise = np.array([120, 100, 90])[None] + rng.normal(0, 0.5, (80, 3))
    snr2, _ = rppg.band_snr(rppg.pos(noise, fps), fps)
    assert snr > snr2


def _ctx(pulse_amp, seed):
    T, fps = 80, 25
    rng = np.random.default_rng(seed)
    pulse = pulse_amp * np.sin(2 * np.pi * 1.2 * np.arange(T) / fps)
    base = np.array([90, 100, 120], np.float32)  # BGR
    frames = np.stack([
        np.clip(base + pulse[t] * np.array([0.5, 0.8, 0.3]) + rng.normal(0, 0.5, (360, 640, 3)), 0, 255).astype(np.uint8)
        for t in range(T)
    ])
    return SimpleNamespace(frames=frames, lms=fake_lms(T), face_ok=np.ones(T, bool), sampled=np.arange(0, T, 5), fps=fps)


def test_score_detects_pulse():
    live, flat = rppg.score(_ctx(2.0, 0)), rppg.score(_ctx(0.0, 1))
    assert set(live) == set(rppg.KEYS)
    assert live["rp_agree"] <= 0.2
    assert live["rp_snr"] > flat["rp_snr"]


def test_score_achromatic_frames_is_nan():
    T, fps = 80, 25
    rng = np.random.default_rng(0)
    gray = np.clip(128 + 20 * np.sin(2 * np.pi * 1.2 * np.arange(T) / fps), 0, 255).astype(np.uint8)
    frames = np.stack([np.full((360, 640, 3), g, np.uint8) for g in gray])
    ctx = SimpleNamespace(frames=frames, lms=fake_lms(T), face_ok=np.ones(T, bool),
                          sampled=np.arange(0, T, 5), fps=fps)
    out = rppg.score(ctx)
    assert set(out) == set(rppg.KEYS)
    assert all(np.isnan(v) for v in out.values())
