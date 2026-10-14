import cv2
import numpy as np

from fdemo.attacks import injection, replay, static_print
from fdemo.signals.spectrum import spectrum_features
from fdemo.signals.temporal import dup_features, noise_corr
from helpers import smooth_frames


def center(frame):
    h, w = frame.shape[:2]
    return cv2.cvtColor(frame[h // 2 - 64:h // 2 + 64, w // 2 - 64:w // 2 + 64], cv2.COLOR_BGR2GRAY).astype(np.float32)


def test_replay_adds_spectral_peaks():
    fr = smooth_frames(4)
    out = replay.generate(fr, np.random.default_rng(3))
    assert out.shape == fr.shape and out.dtype == np.uint8
    assert spectrum_features(center(out[0]))["sp_peaks"] > spectrum_features(center(fr[0]))["sp_peaks"]


def test_resample_creates_duplicates():
    fr = np.stack([np.random.default_rng(i).integers(0, 255, (90, 160, 3), dtype=np.uint8) for i in range(80)])
    assert 0.35 <= dup_features(injection.resample_fps(fr))["tm_dup_rate"] <= 0.45


def test_resample_fps_does_not_slow_down_real_time():
    # a 25 fps real-time clip must still span its full real-time duration: the last emitted
    # tick must come from near the end of the source, not frame ~47/80 (slowed-down time).
    out = injection.resample_fps(np.arange(80))
    assert out[-1] >= 75


def test_resample_fps_output_is_non_decreasing():
    out = injection.resample_fps(np.arange(80))
    assert np.all(np.diff(out) >= 0)


def test_temporal_smooth_raises_noise_correlation():
    fr = np.stack([np.random.default_rng(i).normal(128, 8, (360, 640, 3)).clip(0, 255).astype(np.uint8) for i in range(10)])
    bg = np.ones((360, 640), bool)
    assert noise_corr(injection.temporal_smooth(fr, 0.6), bg) > noise_corr(fr, bg) + 0.2


def test_injection_generate_keeps_length(tmp_path):
    fr = smooth_frames(20)
    out = injection.generate(fr, np.random.default_rng(0))
    assert out.shape == fr.shape


def test_static_print_moves_only_slightly():
    fr = smooth_frames(20)
    out = static_print.generate(fr, np.random.default_rng(0))
    assert out.shape == fr.shape
    assert np.abs(np.diff(out.astype(np.float32), axis=0)).mean() < 10
