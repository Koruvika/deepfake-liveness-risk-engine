from types import SimpleNamespace

import numpy as np

from fdemo.signals import temporal
from helpers import fake_lms, noisy_frame


def test_dup_rate_counts_repeated_frames():
    base = [noisy_frame(20, i, (90, 160)) for i in range(40)]
    d = temporal.dup_features(np.stack([base[i // 2] for i in range(80)]))
    assert 0.45 <= d["tm_dup_rate"] <= 0.55
    assert temporal.dup_features(np.stack(base))["tm_dup_rate"] == 0.0


def test_noise_corr_iid_vs_smoothed():
    iid = np.stack([noisy_frame(8, i) for i in range(20)]).astype(np.float32)
    sm = iid.copy()
    for t in range(1, len(sm)):
        sm[t] = 0.5 * iid[t] + 0.5 * sm[t - 1]
    bg = np.ones((360, 640), bool)
    assert abs(temporal.noise_corr(iid.astype(np.uint8), bg)) < 0.1
    assert temporal.noise_corr(sm.astype(np.uint8), bg) > 0.3


def test_landmark_jitter_smooth_vs_noisy():
    t = np.arange(80)[:, None, None]
    base = np.random.default_rng(0).uniform(200, 400, (1, 478, 2))
    smooth = base + 5 * np.sin(t / 10.0)
    noisy = smooth + np.random.default_rng(1).normal(0, 1.5, smooth.shape)
    assert temporal.landmark_jitter(noisy) > 5 * temporal.landmark_jitter(smooth)


def test_landmark_jitter_ignores_gap_with_pose_jump():
    t = np.arange(80)[:, None, None]
    base = np.random.default_rng(0).uniform(200, 400, (1, 478, 2))
    smooth = base + 5 * np.sin(t / 10.0)
    gapped = smooth.copy()
    ok = np.ones(80, bool)
    ok[30:40] = False
    gapped[30:40] = np.nan
    gapped[40:] += 200.0  # large pose jump straight after the gap
    j_ref = temporal.landmark_jitter(smooth)
    j_gap = temporal.landmark_jitter(gapped, ok)
    assert j_gap <= 1.5 * j_ref


def test_score_keys():
    T = 12
    ctx = SimpleNamespace(frames=np.stack([noisy_frame(5, i) for i in range(T)]), lms=fake_lms(T),
                          face_ok=np.ones(T, bool), sampled=np.arange(T))
    out = temporal.score(ctx)
    assert set(out) == set(temporal.KEYS) and all(np.isfinite(v) for v in out.values())
