import numpy as np
import pandas as pd
import pytest

from fdemo import config
from fdemo.attacks import faceswap_classic as fc
from fdemo.face.geometry import FACE_OVAL, SWAP_POINTS, hull_mask
from helpers import circle_points, noisy_frame


def pseudo_points(seed=0, cx=320, cy=180):
    rng = np.random.default_rng(seed)
    oval = circle_points(cx, cy, 80, len(FACE_OVAL))
    a, r = rng.uniform(0, 2 * np.pi, len(SWAP_POINTS) - len(FACE_OVAL)), rng.uniform(0, 50, len(SWAP_POINTS) - len(FACE_OVAL))
    inner = np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)
    return np.concatenate([oval, inner]).astype(np.float32)


def stripes():
    y = np.indices((360, 640))[0]
    g = (128 + 60 * np.sign(np.sin(2 * np.pi * y / 8))).astype(np.uint8)
    return np.repeat(g[..., None], 3, 2)


def lag_corr(region, lag=8):
    r = region.astype(np.float32).mean(axis=2)
    a, b = r[:-lag].ravel(), r[lag:].ravel()
    return float(np.corrcoef(a, b)[0, 1])


def test_delaunay_indices_valid():
    pts = pseudo_points()
    tris = fc.delaunay(pts, (360, 640))
    assert len(tris) > 50
    assert all(len(set(t)) == 3 and max(t) < len(pts) for t in tris)


@pytest.mark.parametrize("mode", ["clone", "alpha"])
def test_swap_frame_transfers_source_texture(mode):
    src_pts = pseudo_points(0)
    dst_pts = src_pts + np.float32([6, 4])
    target = noisy_frame(20, 0)
    tris = fc.delaunay(dst_pts, target.shape)
    out = fc.swap_frame(target, stripes(), src_pts, dst_pts, tris, mode, 11)
    assert lag_corr(out[150:210, 290:350]) > 0.5
    assert abs(lag_corr(target[150:210, 290:350])) < 0.2
    assert np.array_equal(out[:20, :20], target[:20, :20])


@pytest.mark.parametrize("mode", ["clone", "alpha"])
def test_swap_frame_uniform_images_keeps_pixels_near_target(mode):
    target = np.full((360, 640, 3), 120, np.uint8)
    source = np.full((360, 640, 3), 120, np.uint8)
    pts = pseudo_points(0)
    tris = fc.delaunay(pts, target.shape)
    out = fc.swap_frame(target, source, pts, pts, tris, mode, 21)
    assert np.abs(out.astype(int) - target.astype(int)).max() <= 3


def test_blend_does_not_mutate_mask():
    m = hull_mask(circle_points(), (360, 640))
    before = m.copy()
    fc.blend(noisy_frame(5, 0), noisy_frame(5, 1), m, "clone", 11)
    assert np.array_equal(m, before)


@pytest.mark.models
@pytest.mark.data
def test_generate_on_real_clip():
    from fdemo.io.video import read_frames

    bona = pd.read_csv(config.BONA_CSV)
    a, b = bona.iloc[0], bona[bona.actor != bona.iloc[0].actor].iloc[0]
    target = read_frames(config.BONA_RAW / f"{a.vid}.mp4", 5)
    source = read_frames(config.BONA_RAW / f"{b.vid}.mp4", 1)[0]
    out = fc.generate(target, np.random.default_rng(0), source)
    assert out.shape == target.shape
    assert np.abs(out.astype(int) - target.astype(int)).mean() > 1.0
