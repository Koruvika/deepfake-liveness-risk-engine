import json

import numpy as np

from fdemo import config
from fdemo.data.prepare import bona_id, fetch_ravdess, is_video_only_speech, make_splits, trim_clip
from fdemo.io.video import probe, write_frames


def test_make_splits_balanced_and_deterministic():
    folds = make_splits(config.ACTORS, 4, 1337)
    assert folds == make_splits(config.ACTORS, 4, 1337)
    for f in range(4):
        members = [a for a, k in folds.items() if k == f]
        assert len(members) == 3
        assert any(a % 2 == 1 for a in members) and any(a % 2 == 0 for a in members)


def test_is_video_only_speech():
    assert is_video_only_speech("Actor_01/02-01-03-01-01-01-01.mp4")
    assert not is_video_only_speech("Actor_01/01-01-03-01-01-01-01.mp4")
    assert not is_video_only_speech("Actor_01/02-02-03-01-01-01-01.mp4")


def test_bona_id():
    assert bona_id(3, 7) == "bona_a03_c07"


def test_with_retry_recovers_then_gives_up(monkeypatch):
    import pytest

    from fdemo.data import prepare

    monkeypatch.setattr(prepare.time, "sleep", lambda s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise IOError("504")
        return "ok"

    assert prepare.with_retry(flaky) == "ok" and len(calls) == 3
    with pytest.raises(RuntimeError):
        prepare.with_retry(lambda: (_ for _ in ()).throw(IOError("504")), tries=2)


def test_extract_member_retries_with_fresh_zip(monkeypatch):
    from fdemo.data import prepare

    monkeypatch.setattr(prepare.time, "sleep", lambda s: None)
    instances = []

    class FakeZip:
        def __init__(self, url, timeout=None):
            instances.append(self)
            self.calls = 0

        def extract(self, name, td):
            self.calls += 1
            if len(instances) < 2:
                raise IOError("connection reset")
            return "ok"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(prepare, "RemoteZip", FakeZip)
    assert prepare.with_retry(lambda: prepare.extract_member("u", "n", "/tmp")) == "ok"
    assert len(instances) == 2


def test_fetch_ravdess_folds_cover_all_actors_even_for_actor_subset(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SPLITS", tmp_path / "splits.json")
    monkeypatch.setattr(config, "BONA_CSV", tmp_path / "bona.csv")

    fetch_ravdess(actors=[], seed=1337)

    written = json.loads(config.SPLITS.read_text())["folds"]
    assert sorted(int(a) for a in written) == config.ACTORS
    for f in range(config.N_FOLDS):
        assert sum(1 for k in written.values() if k == f) == 3


def test_trim_clip_produces_80_frames(tmp_path):
    src = tmp_path / "src.mp4"
    frames = np.random.default_rng(0).integers(0, 255, (120, 360, 640, 3), dtype=np.uint8)
    write_frames(frames, src, fps=30)
    dst = tmp_path / "dst.mp4"
    trim_clip(src, dst)
    info = probe(dst)
    assert info["frames"] == config.N_FRAMES and (info["width"], info["height"]) == (640, 360)
