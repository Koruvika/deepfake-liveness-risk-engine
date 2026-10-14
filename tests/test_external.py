import huggingface_hub

from fdemo import config
from fdemo.data import external


def test_fetch_external_isolates_failing_video(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "VIDEOS", tmp_path / "videos")
    monkeypatch.setattr(config, "EXT_SRC", tmp_path / "external_src")
    monkeypatch.setattr(config, "EXT_CSV", tmp_path / "external.csv")
    monkeypatch.setattr(external, "probe", lambda dst: {"frames": config.N_FRAMES})

    g = external.GENERATORS[0]
    files = [f"fake/{g}/test/000.mp4", f"fake/{g}/test/001.mp4"]

    class FakeApi:
        def list_repo_files(self, repo, repo_type=None):
            return files

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download",
        lambda repo, f, repo_type=None, local_dir=None: str(tmp_path / "src.mp4"),
    )

    calls = {"n": 0}

    def fake_capture(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("ffmpeg boom")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"x")

    monkeypatch.setattr(external, "ext_capture", fake_capture)

    df = external.fetch_external(per_gen=2)
    assert len(df) == 1
    assert df.iloc[0].vid == f"ext_{g}_01"
    assert config.EXT_CSV.exists()
    ext_dir = config.VIDEOS / "EXT"
    assert not list(ext_dir.glob("*.part.mp4"))
