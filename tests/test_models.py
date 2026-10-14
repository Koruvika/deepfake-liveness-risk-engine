import numpy as np
import pytest

from fdemo import models
from fdemo.face.align import ARCFACE_DST

pytestmark = pytest.mark.models


def test_minifas_sessions():
    sessions = models.minifas()
    assert [s for _, s in sessions] == [2.7, 4.0]
    for sess, _ in sessions:
        out = sess.run(None, {sess.get_inputs()[0].name: np.zeros((1, 3, 80, 80), np.float32)})[0]
        assert out.shape == (1, 3)


def test_inswapper_emap():
    sess, emap = models.inswapper()
    assert emap.shape == (512, 512)
    assert [i.name for i in sess.get_inputs()] == ["target", "source"]


def test_sbi_forward():
    import torch
    with torch.no_grad():
        y = models.sbi()(torch.zeros(1, 3, 380, 380))
    assert tuple(y.shape) == (1, 2)


def test_arcface_unit_embedding():
    img = np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8)
    e = models.arcface().embed(img, ARCFACE_DST + 40)
    assert e.shape == (512,) and abs(np.linalg.norm(e) - 1) < 1e-4


def test_detectors_return_none_on_blank():
    blank = np.zeros((360, 640, 3), np.uint8)
    assert models.detector()(blank) is None
    assert models.landmarker()(blank) is None


def test_detector_session_uses_configured_threads():
    from fdemo import config

    d = models.detector()
    assert d._net.session.get_session_options().intra_op_num_threads == config.THREADS
    assert d(np.zeros((360, 640, 3), np.uint8)) is None
