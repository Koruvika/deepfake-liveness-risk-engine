from types import SimpleNamespace

import numpy as np

from fdemo.signals import spectrum
from helpers import fake_lms, noisy_frame


def test_grating_creates_spectral_peaks():
    rng = np.random.default_rng(0)
    g = rng.normal(128, 10, (128, 128))
    y, x = np.indices(g.shape)
    grat = g + 20 * np.sin(2 * np.pi * 0.2 * (x * np.cos(0.5) + y * np.sin(0.5)))
    a, b = spectrum.spectrum_features(g), spectrum.spectrum_features(grat)
    assert b["sp_peaks"] > a["sp_peaks"] + 0.1


def test_face_crop_gray_shape():
    crop = spectrum.face_crop_gray(noisy_frame(5, 0), fake_lms(1)[0])
    assert crop.shape == (128, 128) and crop.dtype == np.float32


def test_face_crop_gray_rejects_non_square_edge_clip():
    lms = np.array([[0.0, 0.0], [5.0, 300.0]], np.float32)
    assert spectrum.face_crop_gray(noisy_frame(5, 0), lms) is None


def test_score_keys():
    T = 3
    ctx = SimpleNamespace(frames=np.stack([noisy_frame(5, i) for i in range(T)]), lms=fake_lms(T),
                          face_ok=np.ones(T, bool), sampled=np.arange(T))
    out = spectrum.score(ctx)
    assert set(out) == set(spectrum.KEYS) and all(np.isfinite(v) for v in out.values())
