import numpy as np

from fdemo.ctx import VideoCtx
from fdemo.signals import SIGNALS, SIGNAL_PREFIX
from fdemo.signals.deepfake_sbi import sbi_box
from fdemo.signals.pad_minifasnet import minifas_box, real_prob, valid_bbox


def test_minifas_box_matches_reference_crop():
    # yakhyo AntiSpoofingONNX._crop_face with bbox (100,100,200,200) on a 400x400 image, scale 2.7
    assert minifas_box((100, 100, 200, 200), 400, 400, 2.7) == (15, 15, 285, 285)


def test_minifas_box_clips_scale_to_image():
    x0, y0, x1, y1 = minifas_box((10, 10, 390, 390), 400, 400, 4.0)
    assert x0 >= 0 and y0 >= 0 and x1 <= 399 and y1 <= 399


def test_sbi_box_matches_reference_margins():
    # SBI crop_face(phase='test'): margins w/8 and h/8 on each side
    assert sbi_box((100, 100, 200, 200), 400, 400) == (87, 87, 213, 213)


def test_videoctx_no_face_flag():
    ctx = VideoCtx("v", np.zeros((4, 8, 8, 3), np.uint8), np.zeros((4, 478, 2), np.float32),
                   np.array([True, False, False, False]), np.arange(4))
    assert ctx.no_face


def test_valid_bbox_rejects_zero_size():
    assert valid_bbox((0, 0, 10, 10))
    assert not valid_bbox((0, 0, 1, 10))
    assert not valid_bbox((0, 0, 10, 1))
    assert not valid_bbox((5, 5, 5, 5))


def test_real_prob_never_raises_on_degenerate_bbox():
    img = np.zeros((50, 50, 3), np.uint8)
    p = real_prob(img, (10, 10, 10, 10))
    assert np.isnan(p)


def test_registry_prefixes():
    assert set(SIGNALS) == set(SIGNAL_PREFIX) == {"pad", "sbi", "blend", "spectrum", "temporal", "identity", "rppg"}
