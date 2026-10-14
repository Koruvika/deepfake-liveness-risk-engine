import subprocess

import joblib
import pandas as pd
import pytest
from typer.testing import CliRunner

from fdemo import config, scan
from fdemo.cli import app
from fdemo.fusion.failures import nan_free_cols
from fdemo.fusion.model import Scorer
from fdemo.fusion.policy import Thresholds
from synth import synthetic_dataset


def test_help_lists_commands():
    out = CliRunner().invoke(app, ["--help"]).output
    for cmd in ("fetch", "generate", "extract", "bench", "recommend", "scan"):
        assert cmd in out


def _bundle(tmp_path, df, scorer=None):
    scorer = scorer or Scorer("ALL").fit(df)
    bundle = {"scorer": scorer, "thresholds": Thresholds(0.3, 0.7),
              "medians": df[scorer.cols].median().to_dict(), "columns": scorer.cols,
              "nan_free_cols": nan_free_cols(df, scorer.cols)}
    path = tmp_path / "b.joblib"
    joblib.dump(bundle, path)
    return path


def test_scan_video_with_stubbed_extraction(tmp_path, monkeypatch):
    df = synthetic_dataset(2)
    path = _bundle(tmp_path, df)
    monkeypatch.setattr(scan.config, "BUNDLE", path)
    monkeypatch.setattr(scan, "sdk_capture", lambda src, dst: dst)
    attack_row = df[df.label == 1].iloc[0].to_dict()
    monkeypatch.setattr(scan, "extract_video", lambda vid, p, cache_dir=None: {**attack_row, "no_face": False})
    vendor = tmp_path / "v.json"
    vendor.write_text('[{"externalDatabaseRefID": "u1", "result": {"livenessProven": true}}]')
    out = scan.scan_video(tmp_path / "x.mp4", vendor_json=vendor)
    assert out["decision"] in ("allow", "review", "deny") and 0 <= out["risk"] <= 1
    assert len(out["top_signals"]) == 3 and out["vendor"]["liveness_proven"] is True


def test_scan_missing_bundle_raises_system_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(scan.config, "BUNDLE", tmp_path / "does-not-exist.joblib")
    with pytest.raises(SystemExit, match="model bundle not found"):
        scan.scan_video(tmp_path / "x.mp4")


def test_scan_no_face_routes_to_review_regardless_of_risk(tmp_path, monkeypatch):
    df = synthetic_dataset(2)
    path = _bundle(tmp_path, df)
    monkeypatch.setattr(scan.config, "BUNDLE", path)
    monkeypatch.setattr(scan, "sdk_capture", lambda src, dst: dst)
    attack_row = df[df.label == 1].iloc[0].to_dict()
    # even with a high-risk-looking row, no_face must win and route to review (policy order: no_face before
    # risk_high/risk_medium).
    monkeypatch.setattr(scan, "extract_video", lambda vid, p, cache_dir=None: {**attack_row, "no_face": True})
    out = scan.scan_video(tmp_path / "x.mp4")
    assert out["decision"] == "review" and out["reason"] == "no_face"


def test_scan_decode_failure_raises_system_exit(tmp_path, monkeypatch):
    df = synthetic_dataset(2)
    path = _bundle(tmp_path, df)
    monkeypatch.setattr(scan.config, "BUNDLE", path)

    def _boom(src, dst):
        raise subprocess.CalledProcessError(1, ["ffmpeg"])

    monkeypatch.setattr(scan, "sdk_capture", _boom)
    with pytest.raises(SystemExit, match="could not decode video"):
        scan.scan_video(tmp_path / "x.mp4")


def test_scan_malformed_vendor_json_raises_system_exit(tmp_path, monkeypatch):
    df = synthetic_dataset(2)
    path = _bundle(tmp_path, df)
    monkeypatch.setattr(scan.config, "BUNDLE", path)
    monkeypatch.setattr(scan, "sdk_capture", lambda src, dst: dst)
    attack_row = df[df.label == 1].iloc[0].to_dict()
    monkeypatch.setattr(scan, "extract_video", lambda vid, p, cache_dir=None: {**attack_row, "no_face": False})

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("not json")
    with pytest.raises(SystemExit, match="invalid vendor session JSON"):
        scan.scan_video(tmp_path / "x.mp4", vendor_json=bad_json)

    bad_payload = tmp_path / "bad2.json"
    bad_payload.write_text('[{"externalDatabaseRefID": "a"}, {"externalDatabaseRefID": "b"}]')
    with pytest.raises(SystemExit, match="invalid vendor session JSON"):
        scan.scan_video(tmp_path / "x.mp4", vendor_json=bad_payload)


def test_scan_signal_failure_routes_to_review_regardless_of_risk(tmp_path, monkeypatch):
    df = synthetic_dataset(2)
    path = _bundle(tmp_path, df)
    monkeypatch.setattr(scan.config, "BUNDLE", path)
    monkeypatch.setattr(scan, "sdk_capture", lambda src, dst: dst)
    attack_row = df[df.label == 1].iloc[0].to_dict()
    # a signal that errored must route to review even though the risk score (fabricated via imputation) looks
    # high-risk -- the policy order is no_face, then signal_failure, then risk thresholds.
    monkeypatch.setattr(scan, "extract_video",
                         lambda vid, p, cache_dir=None: {**attack_row, "no_face": False, "err_sbi": 1})
    out = scan.scan_video(tmp_path / "x.mp4")
    assert out["decision"] == "review" and out["reason"] == "signal_failure"


def test_scan_nan_in_column_that_had_nan_during_training_is_not_signal_failure(tmp_path, monkeypatch):
    """A NaN the model's imputer was already trained to handle (the column had NaN in training) must not
    trip signal_failure -- only a NaN in a column that was always populated at training time is suspicious."""
    df = synthetic_dataset(5)
    flaky_col = Scorer("ALL").fit(df).cols[0]
    df.loc[df.index[0], flaky_col] = float("nan")  # one NaN in training -> flaky_col excluded from nan_free_cols
    scorer = Scorer("ALL").fit(df)
    assert flaky_col not in nan_free_cols(df, scorer.cols)
    path = _bundle(tmp_path, df, scorer=scorer)
    monkeypatch.setattr(scan.config, "BUNDLE", path)
    monkeypatch.setattr(scan, "sdk_capture", lambda src, dst: dst)
    attack_row = df[df.label == 1].iloc[0].to_dict()
    attack_row[flaky_col] = float("nan")
    monkeypatch.setattr(scan, "extract_video", lambda vid, p, cache_dir=None: {**attack_row, "no_face": False})
    out = scan.scan_video(tmp_path / "x.mp4")
    assert out["reason"] != "signal_failure"


def test_scan_nan_in_nan_free_column_is_signal_failure(tmp_path, monkeypatch):
    """A NaN in a column that was always populated during training means the model never learned a fallback
    for it -- must route to review/signal_failure."""
    df = synthetic_dataset(5)
    scorer = Scorer("ALL").fit(df)
    nfc = nan_free_cols(df, scorer.cols)
    assert nfc  # synthetic_dataset has no NaNs, so every scorer column is nan_free
    path = _bundle(tmp_path, df, scorer=scorer)
    monkeypatch.setattr(scan.config, "BUNDLE", path)
    monkeypatch.setattr(scan, "sdk_capture", lambda src, dst: dst)
    attack_row = df[df.label == 1].iloc[0].to_dict()
    attack_row[nfc[0]] = float("nan")
    monkeypatch.setattr(scan, "extract_video", lambda vid, p, cache_dir=None: {**attack_row, "no_face": False})
    out = scan.scan_video(tmp_path / "x.mp4")
    assert out["decision"] == "review" and out["reason"] == "signal_failure"


def test_scan_passes_expected_ref_to_vendor_parsing(tmp_path, monkeypatch):
    df = synthetic_dataset(2)
    path = _bundle(tmp_path, df)
    monkeypatch.setattr(scan.config, "BUNDLE", path)
    monkeypatch.setattr(scan, "sdk_capture", lambda src, dst: dst)
    attack_row = df[df.label == 1].iloc[0].to_dict()
    monkeypatch.setattr(scan, "extract_video", lambda vid, p, cache_dir=None: {**attack_row, "no_face": False})
    vendor = tmp_path / "v.json"
    vendor.write_text('[{"externalDatabaseRefID": "u1", "result": {"livenessProven": true}}]')
    with pytest.raises(SystemExit, match="invalid vendor session JSON"):
        scan.scan_video(tmp_path / "x.mp4", vendor_json=vendor, expected_ref="other-ref")
    out = scan.scan_video(tmp_path / "x.mp4", vendor_json=vendor, expected_ref="u1")
    assert out["vendor"]["external_ref_id"] == "u1"


def test_cli_scan_rejects_missing_video_file(tmp_path):
    result = CliRunner().invoke(app, ["scan", str(tmp_path / "missing.mp4")])
    assert result.exit_code != 0


def test_cli_scan_rejects_missing_vendor_file(tmp_path):
    video = tmp_path / "x.mp4"
    video.write_bytes(b"")
    result = CliRunner().invoke(app, ["scan", str(video), "--vendor", str(tmp_path / "missing.json")])
    assert result.exit_code != 0


def test_cli_scan_rejects_expected_ref_without_vendor(tmp_path):
    video = tmp_path / "x.mp4"
    video.write_bytes(b"")
    result = CliRunner().invoke(app, ["scan", str(video), "--expected-ref", "u1"])
    assert result.exit_code != 0
    assert "--expected-ref requires --vendor" in str(result.output) + str(result.exception)


def test_cli_generate_calls_recapture_bona_before_planning(tmp_path, monkeypatch):
    import fdemo.attacks.runner as runner

    bona = pd.DataFrame({"vid": ["bona_a01_c00"], "actor": [1], "fold": [0], "source_name": ["x"]})
    bona_csv = tmp_path / "bona.csv"
    bona.to_csv(bona_csv, index=False)
    monkeypatch.setattr(config, "BONA_CSV", bona_csv)

    calls = []
    monkeypatch.setattr(runner, "recapture_bona", lambda df, workers=config.WORKERS: calls.append(list(df.vid)))
    monkeypatch.setattr(runner, "plan_jobs", lambda df: [])
    monkeypatch.setattr(runner, "generate_all", lambda jobs, workers=config.WORKERS: [])
    monkeypatch.setattr(runner, "build_manifest", lambda bona, rows, ext: pd.DataFrame())

    result = CliRunner().invoke(app, ["generate", "--workers", "0"])
    assert result.exit_code == 0, result.output
    assert calls == [["bona_a01_c00"]]
