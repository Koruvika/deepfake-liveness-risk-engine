import numpy as np
import pandas as pd
import pytest

from fdemo import config
from fdemo.attacks.faceswap_neural import paste_back


def test_paste_back_places_face_and_feathers():
    frame = np.zeros((360, 640, 3), np.uint8)
    face = np.full((128, 128, 3), 255, np.uint8)
    M = np.float32([[1, 0, -200], [0, 1, -100]])  # image -> crop: crop covers x 200..328, y 100..228
    out = paste_back(frame, face, M)
    assert out[164, 264].mean() > 240
    assert out[10, 10].sum() == 0
    assert out[102, 264].mean() < out[164, 264].mean()  # edge is feathered (eroded + blurred mask)


@pytest.mark.models
@pytest.mark.data
def test_swap_moves_identity_towards_source():
    from fdemo import models
    from fdemo.attacks.faceswap_neural import generate
    from fdemo.io.video import read_frames

    bona = pd.read_csv(config.BONA_CSV)
    a, b = bona.iloc[0], bona[bona.actor != bona.iloc[0].actor].iloc[0]
    target = read_frames(config.BONA_RAW / f"{a.vid}.mp4", 2)
    source = read_frames(config.BONA_RAW / f"{b.vid}.mp4", 1)[0]
    out = generate(target, np.random.default_rng(0), source)
    det, arc = models.detector(), models.arcface()
    e = lambda img: arc.embed(img, det(img).kps)
    assert float(e(out[0]) @ e(source)) > float(e(out[0]) @ e(target[0]))
