from pathlib import Path

import numpy as np
import pandas as pd

from fdemo import config
from fdemo.attacks.runner import (
    Job, _write_and_capture, bona_rows, capture_like_attack, manifest_row, pick_source_frame, plan_jobs,
    recapture_bona, run_job, run_job_safe, split_counts,
)
from fdemo.data.external import GENERATORS, select_files
from fdemo.data.prepare import bona_id, make_splits
from fdemo.io.video import read_frames, write_frames


def fake_bona():
    folds = make_splits(config.ACTORS, config.N_FOLDS, config.SEED)
    return pd.DataFrame([
        {"vid": bona_id(a, k), "actor": a, "fold": folds[a], "source_name": f"x{k}"}
        for a in config.ACTORS for k in range(config.CLIPS_PER_ACTOR)
    ])


def test_split_counts():
    assert split_counts(150, 4) == [38, 38, 37, 37]


def test_plan_jobs_counts_and_folds():
    bona = fake_bona()
    jobs = plan_jobs(bona)
    assert jobs == plan_jobs(bona)
    fold_of = dict(zip(bona.actor, bona.fold))
    for attack in config.ATTACKS:
        js = [j for j in jobs if j.attack == attack]
        assert len(js) == config.N_ATTACK
        assert len({j.target_vid for j in js}) == config.N_ATTACK
        assert sorted(pd.Series([j.fold for j in js]).value_counts().tolist(), reverse=True) == [38, 38, 37, 37]
    for j in jobs:
        assert fold_of[j.target_actor] == j.fold
        if j.attack in ("A1", "A2"):
            assert j.source_actor != j.target_actor and fold_of[j.source_actor] == j.fold


def test_a4_mirrors_a2():
    jobs = plan_jobs(fake_bona())
    a2 = {j.vid: j for j in jobs if j.attack == "A2"}
    a4 = [j for j in jobs if j.attack == "A4"]
    assert all(j.source_vid in a2 and a2[j.source_vid].target_vid == j.target_vid for j in a4)


def test_manifest_rows():
    j = plan_jobs(fake_bona())[0]
    r = manifest_row(j)
    assert r["label"] == 1 and r["group"] == config.ATTACK_GROUP[j.attack] and r["path"].endswith(f"{j.vid}.mp4")
    b = bona_rows(fake_bona()).iloc[0]
    assert b["label"] == 0 and b["attack"] == "bona" and b["source_actor"] == -1


def test_select_files():
    files = [f"fake/{g}/test/{i:03d}.mp4" for g in GENERATORS for i in range(30)] + ["fake/Hallo/train/x.mp4"]
    sel = select_files(files, 20)
    assert set(sel) == set(GENERATORS) and all(len(v) == 20 for v in sel.values())
    assert all("/test/" in f for v in sel.values() for f in v)


def test_pick_source_frame_skips_faceless(monkeypatch):
    from fdemo import models
    from fdemo.attacks import runner

    frames = np.arange(8)  # frame "i" is represented by the int i for identity checks
    monkeypatch.setattr(models, "detector", lambda: (lambda f: "face" if f == 7 else None))
    assert pick_source_frame(frames) == 7


def test_pick_source_frame_raises_if_no_face(monkeypatch):
    from fdemo import models

    frames = np.arange(8)
    monkeypatch.setattr(models, "detector", lambda: (lambda f: None))
    import pytest

    with pytest.raises(ValueError):
        pick_source_frame(frames)


def test_run_job_safe_returns_none_on_exception(monkeypatch):
    from fdemo.attacks import runner

    def boom(job):
        raise RuntimeError("x")

    monkeypatch.setattr(runner, "run_job", boom)
    job = plan_jobs(fake_bona())[0]
    assert run_job_safe(job) is None


def test_run_job_a4_missing_source_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VIDEOS", tmp_path / "videos")
    monkeypatch.setattr(config, "ATTACK_RAW", tmp_path / "attack_raw")
    job = Job("A4_000", "A4", "bona_a01_c00", 1, 0, "A2_missing", 2, 123)
    assert run_job(job) is None


def test_run_job_atomic_output(tmp_path, monkeypatch):
    from fdemo.attacks import runner

    monkeypatch.setattr(config, "VIDEOS", tmp_path / "videos")
    monkeypatch.setattr(config, "BONA_RAW", tmp_path / "bona_raw")
    monkeypatch.setattr(runner, "read_frames", lambda path: np.zeros((4, 4, 4, 3), np.uint8))

    def fake_write_frames(frames, path, crf=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        return path

    def fake_sdk_capture(src, dst):
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"y")
        return dst

    monkeypatch.setattr(runner, "write_frames", fake_write_frames)
    monkeypatch.setattr(runner, "sdk_capture", fake_sdk_capture)
    monkeypatch.setattr(runner, "_generators", lambda: {"A3": lambda target, rng, source: target})

    job = Job("A3_000", "A3", "bona_a01_c00", 1, 0, None, None, 7)
    row = run_job(job)
    final = config.VIDEOS / "A3" / "A3_000.mp4"
    assert final.exists()
    assert not list(final.parent.glob("*.part.mp4"))
    assert row["vid"] == "A3_000"


def _synthetic_raw(path):
    frames = np.full((20, 360, 640, 3), 90, np.uint8)
    for t in range(20):
        frames[t, 100:160, 50 + 5 * t:110 + 5 * t] = 220
    write_frames(frames, path, fps=config.FPS)
    return frames


def test_capture_like_attack_matches_write_and_capture(tmp_path):
    """Bona fide (capture_like_attack) and a no-op attack (_write_and_capture on the same raw frames) must
    be byte-path identical -- the 'null attack' is exactly what every attack does."""
    raw = tmp_path / "raw.mp4"
    _synthetic_raw(raw)

    a = capture_like_attack(raw, tmp_path / "bona_final.mp4")
    b = _write_and_capture(read_frames(raw), tmp_path / "attack_final.mp4")

    fa, fb = read_frames(a), read_frames(b)
    assert fa.shape == fb.shape
    assert np.abs(fa.astype(int) - fb.astype(int)).mean() < 0.5


def test_recapture_bona_writes_marker_and_skips_on_second_call(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VIDEOS", tmp_path / "videos")
    monkeypatch.setattr(config, "BONA_RAW", tmp_path / "bona_raw")

    bona = pd.DataFrame({"vid": ["bona_a01_c00", "bona_a01_c01"]})
    for vid in bona.vid:
        _synthetic_raw(config.BONA_RAW / f"{vid}.mp4")

    recapture_bona(bona, workers=0)

    marker = config.VIDEOS / "bona" / ".null_attack_v1"
    assert marker.exists()
    for vid in bona.vid:
        assert (config.VIDEOS / "bona" / f"{vid}.mp4").exists()

    calls = []
    import fdemo.attacks.runner as runner_mod

    monkeypatch.setattr(runner_mod, "capture_like_attack", lambda raw, final: calls.append(final))
    recapture_bona(bona, workers=0)
    assert calls == []  # skipped entirely: marker already present
