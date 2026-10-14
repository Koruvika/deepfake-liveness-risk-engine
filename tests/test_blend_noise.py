from types import SimpleNamespace

import numpy as np

from fdemo.face.geometry import hull_mask
from fdemo.signals import blend_noise
from fdemo.signals.base import topk_mean
from helpers import circle_points, noisy_frame


def test_topk_mean():
    assert topk_mean([1, 5, 3, 4], 2) == 4.5
    assert np.isnan(topk_mean([], 3))


def test_topk_mean_ignores_non_finite():
    assert topk_mean([1, float("nan"), 5, 3], 2) == 4.0
    assert np.isnan(topk_mean([float("nan"), float("inf")], 2))


def test_pasted_low_noise_region_is_detected():
    pts = circle_points()
    plain = noisy_frame(6, 0)
    pasted = plain.copy()
    m = hull_mask(pts, plain.shape) > 0
    pasted[m] = noisy_frame(1, 1)[m]
    a = blend_noise.frame_features(plain, pts)
    b = blend_noise.frame_features(pasted, pts)
    assert abs(b["bn_core_bg"]) > abs(a["bn_core_bg"]) + 0.5
    assert b["bn_ring_core"] > a["bn_ring_core"] + 0.5


def test_score_returns_all_keys():
    T = 4
    lms = np.repeat(np.pad(circle_points(n=36), ((0, 442), (0, 0)), mode="edge")[None], T, 0)
    ctx = SimpleNamespace(frames=np.stack([noisy_frame(6, i) for i in range(T)]), lms=lms,
                          face_ok=np.ones(T, bool), sampled=np.arange(T))
    out = blend_noise.score(ctx)
    assert set(out) == set(blend_noise.KEYS)
    assert all(np.isfinite(v) for v in out.values())


def test_score_without_faces_is_nan():
    ctx = SimpleNamespace(frames=np.zeros((2, 360, 640, 3), np.uint8), lms=np.full((2, 478, 2), np.nan),
                          face_ok=np.zeros(2, bool), sampled=np.arange(2))
    assert all(np.isnan(v) for v in blend_noise.score(ctx).values())


def test_flat_background_frame_returns_none():
    pts = circle_points()
    frame = np.full((360, 640, 3), 255, np.uint8)
    noisy = noisy_frame(6, 0)
    m = hull_mask(pts, frame.shape) > 0
    frame[m] = noisy[m]
    assert blend_noise.frame_features(frame, pts) is None


def test_score_tolerates_one_flat_frame_among_noisy():
    pts = circle_points()
    flat = np.full((360, 640, 3), 255, np.uint8)
    m = hull_mask(pts, flat.shape) > 0
    flat[m] = noisy_frame(6, 99)[m]
    frames = [flat] + [noisy_frame(6, i) for i in range(1, 4)]
    lms = np.repeat(np.pad(pts, ((0, 442), (0, 0)), mode="edge")[None], 4, 0)
    ctx = SimpleNamespace(frames=np.stack(frames), lms=lms, face_ok=np.ones(4, bool), sampled=np.arange(4))
    out = blend_noise.score(ctx)
    assert set(out) == set(blend_noise.KEYS)
    assert all(np.isfinite(v) for v in out.values())
