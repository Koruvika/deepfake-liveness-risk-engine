import numpy as np
import pytest

from fdemo.face.align import ARCFACE_DST, estimate_norm, norm_crop
from fdemo.face.geometry import FACE_OVAL, ROIS, SWAP_POINTS, hull_mask, roi_mask
from helpers import circle_points


def test_hull_mask_area_matches_circle():
    m = hull_mask(circle_points(r=50, n=72), (360, 640))
    assert abs((m > 0).sum() - np.pi * 50 ** 2) / (np.pi * 50 ** 2) < 0.03


def test_roi_mask_uses_selected_indices():
    lms = np.zeros((478, 2), np.float32)
    lms[ROIS["forehead"]] = circle_points(100, 100, 20, len(ROIS["forehead"]))
    assert (roi_mask(lms, ROIS["forehead"], (360, 640)) > 0).sum() > 500


def test_swap_points_unique_and_oval_first():
    assert len(SWAP_POINTS) == len(set(SWAP_POINTS))
    assert SWAP_POINTS[: len(FACE_OVAL)] == FACE_OVAL


def test_estimate_norm_is_identity_on_template_112():
    M = estimate_norm(ARCFACE_DST.copy(), 112)
    assert np.allclose(M, [[1, 0, 0], [0, 1, 0]], atol=1e-3)


def test_estimate_norm_128_shifts_by_8px():
    M = estimate_norm(ARCFACE_DST + [8.0, 0.0], 128)
    assert np.allclose(M, [[1, 0, 0], [0, 1, 0]], atol=1e-3)


def test_norm_crop_shape():
    img = np.zeros((360, 640, 3), np.uint8)
    crop, M = norm_crop(img, ARCFACE_DST + [200, 100], 128)
    assert crop.shape == (128, 128, 3) and M.shape == (2, 3)


def test_estimate_norm_raises_on_degenerate_keypoints():
    kps = np.full((5, 2), [100.0, 100.0], np.float32)
    with pytest.raises(ValueError):
        estimate_norm(kps, 112)


def test_roi_mask_accepts_3d_landmarks():
    lms = np.zeros((478, 3), np.float32)
    lms[ROIS["forehead"]] = np.concatenate(
        [circle_points(100, 100, 20, len(ROIS["forehead"])), np.zeros((len(ROIS["forehead"]), 1), np.float32)],
        axis=1,
    )
    assert (roi_mask(lms, ROIS["forehead"], (360, 640)) > 0).sum() > 500
