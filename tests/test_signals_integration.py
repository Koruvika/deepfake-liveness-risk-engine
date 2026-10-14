import numpy as np
import pandas as pd
import pytest

from fdemo import config
from fdemo.ctx import build_ctx
from fdemo.signals import SIGNALS

pytestmark = [pytest.mark.models, pytest.mark.data]


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    vid = pd.read_csv(config.BONA_CSV).iloc[0]["vid"]
    return build_ctx(config.VIDEOS / "bona" / f"{vid}.mp4", vid, cache_dir=tmp_path_factory.mktemp("ctx"))


def test_ctx_finds_face(ctx):
    assert ctx.frames.shape == (config.N_FRAMES, config.HEIGHT, config.WIDTH, 3)
    assert ctx.face_ok.mean() > 0.9


@pytest.mark.parametrize("name", list(SIGNALS))
def test_signal_runs_on_bona(ctx, name):
    out = SIGNALS[name](ctx)
    assert out and all(np.isfinite(v) for v in out.values())


def test_identity_is_stable_for_bona(ctx):
    assert SIGNALS["identity"](ctx)["id_drift_min"] > 0.5
