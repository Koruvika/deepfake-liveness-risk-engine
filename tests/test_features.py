import os
from types import SimpleNamespace

import pandas as pd

from fdemo import config, features


def test_extract_video_isolates_failing_signal(monkeypatch):
    monkeypatch.setattr(features, "build_ctx", lambda path, vid, cache_dir=None: SimpleNamespace(no_face=False))

    def boom(ctx):
        raise RuntimeError("x")

    monkeypatch.setattr(features, "SIGNALS", {"a": lambda ctx: {"a_x": 1.0}, "b": boom})
    row = features.extract_video("v1", "unused.mp4")
    assert row["vid"] == "v1" and row["a_x"] == 1.0 and row["no_face"] is False
    assert "lat_a" in row and "lat_b" in row and "lat_ctx" in row
    assert not any(k.startswith("b_") for k in row)
    assert row["err_b"] == 1


def test_extract_all_skips_done(monkeypatch, tmp_path):
    out = tmp_path / "f.csv"
    pd.DataFrame([{"vid": "done", "a_x": 0.0}]).to_csv(out, index=False)
    monkeypatch.setattr(features, "extract_video", lambda vid, path, cache_dir=None: {"vid": vid, "a_x": 2.0})
    manifest = pd.DataFrame({"vid": ["done", "new"], "path": ["a.mp4", "b.mp4"]})
    df = features.extract_all(manifest, out_csv=out, workers=0)
    assert sorted(df.vid) == ["done", "new"]
    assert pd.read_csv(out).set_index("vid").loc["new", "a_x"] == 2.0


def test_extract_all_isolates_failing_video_workers0(monkeypatch, tmp_path):
    out = tmp_path / "f.csv"

    def fake_extract(vid, path, cache_dir=None):
        if vid == "bad":
            raise RuntimeError("boom")
        return {"vid": vid, "a_x": 1.0}

    monkeypatch.setattr(features, "extract_video", fake_extract)
    manifest = pd.DataFrame({"vid": ["good", "bad"], "path": ["a.mp4", "b.mp4"]})
    df = features.extract_all(manifest, out_csv=out, workers=0)
    assert list(df.vid) == ["good"]
    assert list(pd.read_csv(out).vid) == ["good"]
    assert not out.with_suffix(".tmp").exists()


def test_extract_video_records_src_stat(monkeypatch, tmp_path):
    monkeypatch.setattr(features, "build_ctx", lambda path, vid, cache_dir=None: SimpleNamespace(no_face=False))
    monkeypatch.setattr(features, "SIGNALS", {})
    p = tmp_path / "v.mp4"
    p.write_bytes(b"hello")
    st = p.stat()
    row = features.extract_video("v1", str(p))
    assert row["src_mtime_ns"] == st.st_mtime_ns
    assert row["src_size"] == st.st_size


def test_extract_all_reextracts_rows_missing_src_cols(monkeypatch, tmp_path):
    """Rows cached before src_mtime_ns/src_size existed must be re-extracted on resume."""
    monkeypatch.setattr(config, "ROOT", tmp_path)
    p = tmp_path / "a.mp4"
    p.write_bytes(b"hello")
    out = tmp_path / "f.csv"
    pd.DataFrame([{"vid": "a", "a_x": 0.0}]).to_csv(out, index=False)
    manifest = pd.DataFrame({"vid": ["a"], "path": ["a.mp4"]})

    calls = []

    def fake_extract(vid, path, cache_dir=None):
        calls.append(vid)
        st = os.stat(path)
        return {"vid": vid, "a_x": 1.0, "src_mtime_ns": st.st_mtime_ns, "src_size": st.st_size}

    monkeypatch.setattr(features, "extract_video", fake_extract)
    df = features.extract_all(manifest, out_csv=out, workers=0)
    assert calls == ["a"]
    assert list(df.vid) == ["a"]
    assert df.set_index("vid").loc["a", "a_x"] == 1.0


def test_extract_all_skips_unchanged_source(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    p = tmp_path / "a.mp4"
    p.write_bytes(b"hello")
    st = p.stat()
    out = tmp_path / "f.csv"
    pd.DataFrame([{"vid": "a", "a_x": 0.0, "src_mtime_ns": st.st_mtime_ns,
                  "src_size": st.st_size}]).to_csv(out, index=False)
    manifest = pd.DataFrame({"vid": ["a"], "path": ["a.mp4"]})

    calls = []

    def fake_extract(vid, path, cache_dir=None):
        calls.append(vid)
        return {"vid": vid, "a_x": 1.0}

    monkeypatch.setattr(features, "extract_video", fake_extract)
    df = features.extract_all(manifest, out_csv=out, workers=0)
    assert calls == []
    assert df.set_index("vid").loc["a", "a_x"] == 0.0


def test_extract_all_mixed_empty_and_valid_src_stat_keeps_valid_rows_fresh(monkeypatch, tmp_path):
    """A features.csv with one row missing src_mtime_ns (forcing a naive read to float64, which loses
    precision on ~1.8e18 nanosecond timestamps) must not make an unrelated, genuinely unchanged row look
    stale."""
    monkeypatch.setattr(config, "ROOT", tmp_path)
    a = tmp_path / "a.mp4"
    a.write_bytes(b"hello")
    b = tmp_path / "b.mp4"
    b.write_bytes(b"world")
    st_a = a.stat()
    out = tmp_path / "f.csv"
    pd.DataFrame([
        {"vid": "a", "a_x": 0.0, "src_mtime_ns": st_a.st_mtime_ns, "src_size": st_a.st_size},
        {"vid": "b", "a_x": 9.0, "src_mtime_ns": "", "src_size": ""},
    ]).to_csv(out, index=False)
    manifest = pd.DataFrame({"vid": ["a", "b"], "path": ["a.mp4", "b.mp4"]})

    calls = []

    def fake_extract(vid, path, cache_dir=None):
        calls.append(vid)
        st = os.stat(path)
        return {"vid": vid, "a_x": 1.0, "src_mtime_ns": st.st_mtime_ns, "src_size": st.st_size}

    monkeypatch.setattr(features, "extract_video", fake_extract)
    df = features.extract_all(manifest, out_csv=out, workers=0)
    assert calls == ["b"]  # only the row missing src stat is re-extracted
    assert df.set_index("vid").loc["a", "a_x"] == 0.0
    assert df.set_index("vid").loc["b", "a_x"] == 1.0


def test_extract_all_reextracts_changed_source(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    p = tmp_path / "a.mp4"
    p.write_bytes(b"hello")
    out = tmp_path / "f.csv"
    pd.DataFrame([{"vid": "a", "a_x": 0.0, "src_mtime_ns": 1, "src_size": 999}]).to_csv(out, index=False)
    manifest = pd.DataFrame({"vid": ["a"], "path": ["a.mp4"]})

    def fake_extract(vid, path, cache_dir=None):
        st = os.stat(path)
        return {"vid": vid, "a_x": 2.0, "src_mtime_ns": st.st_mtime_ns, "src_size": st.st_size}

    monkeypatch.setattr(features, "extract_video", fake_extract)
    df = features.extract_all(manifest, out_csv=out, workers=0)
    assert list(df.vid) == ["a"]
    assert df.set_index("vid").loc["a", "a_x"] == 2.0


def test_job_wraps_extract_video_exception_as_error_row():
    def boom(vid, path, cache_dir=None):
        raise RuntimeError("boom")

    import fdemo.features as features_mod

    orig = features_mod.extract_video
    try:
        features_mod.extract_video = boom
        row = features_mod._job(("bad", "p.mp4"))
    finally:
        features_mod.extract_video = orig
    assert row["vid"] == "bad" and "error" in row
