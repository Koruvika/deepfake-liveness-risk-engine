import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fdemo.fusion.policy as policy_mod
from fdemo.eval import bench, recommend, report
from fdemo.eval.recommend import psi
from fdemo.eval.summary import summarize_p2
from synth import synthetic_dataset


def test_md_table_formats():
    md = report.md_table(pd.DataFrame({"a": ["x"], "b": [0.12345], "c": [np.nan]}))
    assert md.splitlines()[0] == "| a | b | c |" and "0.123" in md and "—" in md


def test_psi_zero_for_identical_and_positive_for_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, a) < 1e-9
    assert psi(a, a + 1.0) > 0.2


def _run_bench_on(df, tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "load_dataset", lambda: (df, df.iloc[:0], 0))
    monkeypatch.setattr(bench.config, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(bench.config, "FIGURES", tmp_path / "figures")
    monkeypatch.setattr(bench.config, "REPORTS", tmp_path)
    monkeypatch.setattr(bench.config, "BUNDLE", tmp_path / "bundle.joblib")
    monkeypatch.setattr(bench, "ABLATION_GROUPS", {"pad": "pad_", "blend": "bn_"})
    return bench.run_bench()


def test_bench_end_to_end_on_synthetic(tmp_path, monkeypatch):
    df = synthetic_dataset(3, with_ext=True)
    out = _run_bench_on(df, tmp_path, monkeypatch)
    text = out.read_text()
    for heading in ["## 1. Summary", "## 4. P2", "## 7. How a typical passive liveness SDK responds",
                    "## 8. End-to-end", "## 11. Signal coverage", "## 12. Limitations"]:
        assert heading in text
    assert "actor-bootstrap" in text and "Statistics" in text
    assert "no_face dropped" in text
    for name in ("pad", "sbi", "blend", "spectrum", "temporal", "identity", "rppg"):
        assert name in text
    assert "n_videos" in text
    assert (tmp_path / "figures" / "roc_p1.png").exists() and (tmp_path / "bundle.joblib").exists()

    # Figure/table parity for the P2 APCER bar chart, and the P1 median-threshold caption under the histogram.
    assert "Same data as the figure above" in text
    assert re.search(r"\|\s*attack\s*\|\s*B1\s*\|\s*B2\s*\|\s*ALL\s*\|", text)
    assert "P1 median τ_review" in text and "τ_deny" in text

    # iBeta footnote for small samples.
    assert "fewer than 150 attempts" in text

    # Ablation reuses the already-computed macro and carries a legend.
    assert "removing the group lets more attacks through" in text

    # P3's padded-background capture caveat and the ALL-tm comparison row: padding-reactive signals would
    # make the fakes easier to flag (deflating P3 APCER / overstating detection), not the reverse.
    assert "blurred 16:9 background" in text
    assert "ALL−tm" in text
    assert "deflat" in text.lower()

    # Deployed-vs-validated thresholds: train_final wrote final_thresholds.json before the report text was built.
    assert (tmp_path / "results" / "final_thresholds.json").exists()

    # Numeric cross-check: the §4 macro ACER for ALL matches a fresh summarize_p2 computed from the CSV on disk.
    res = pd.read_csv(tmp_path / "results" / "P2_ALL.csv")
    _, _, macro = summarize_p2(res)
    assert report.pct(macro["acer"]) in text

    rec = recommend.write_recommendations(results_csv=tmp_path / "results" / "P2_ALL.csv",
                                          out=tmp_path / "recommendations.md")
    rtext = rec.read_text()
    assert "## Threshold sweep" in rtext and "minimumMatchLevel3D2DMatching" in rtext
    assert "deployed in `fdemo scan`" in rtext and "validated on P2" in rtext
    deployed = json.loads((tmp_path / "results" / "final_thresholds.json").read_text())
    assert f"{deployed['review']:.3f}" in rtext and f"{deployed['deny']:.3f}" in rtext
    assert "sessions routed to review by the risk band" in rtext


def _toy_e2e():
    rows = []
    for mode in ("vendor-only", "vendor+engine"):
        for attack, allow, review, deny in (("bona", 0.9, 0.05, 0.05), ("A5", 0.1, 0.3, 0.6), ("A2", 0.2, 0.3, 0.5)):
            rows.append({"scenario": "V-bypassed", "mode": mode, "attack": attack, "n": 10, "n_videos": 10,
                        "allow": allow, "review": review, "deny": deny})
    return pd.DataFrame(rows)


def test_plot_e2e_uses_a_single_figure_level_legend(tmp_path, monkeypatch):
    """A legend placed inside the right-hand panel (e.g. lower-right) can overlap a bar (the A5 row); a
    figure-level legend above both panels never can."""
    captured = {}
    orig_finish = report._finish

    def spy(fig, path, title, top=None):
        captured["fig"], captured["top"] = fig, top
        return orig_finish(fig, path, title, top=top)

    monkeypatch.setattr(report, "_finish", spy)
    report.plot_e2e(_toy_e2e(), tmp_path / "e2e.png")
    fig = captured["fig"]
    assert len(fig.legends) == 1
    for ax in fig.axes:
        assert ax.get_legend() is None
    assert captured["top"] is not None and captured["top"] < 1.0


def test_plot_risk_hist_uses_log_yscale_and_axes_fraction_labels(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    res = pd.DataFrame({
        "label": [0] * 50 + [1] * 50,
        "score": np.concatenate([rng.uniform(0, 0.3, 50), [0.97] * 40, rng.uniform(0.3, 0.6, 10)]),
        "tau_review": 0.5, "tau_deny": 0.8,
    })
    captured = {}
    orig_finish = report._finish

    def spy(fig, path, title, top=None):
        captured["fig"] = fig
        return orig_finish(fig, path, title, top=top)

    monkeypatch.setattr(report, "_finish", spy)
    report.plot_risk_hist(res, tmp_path / "hist.png")
    ax = captured["fig"].axes[0]
    assert ax.get_yscale() == "log"
    assert ax.get_ylim()[0] == pytest.approx(0.8)
    assert "log scale" in ax.get_ylabel()
    threshold_texts = [t for t in ax.texts if "τ=" in t.get_text()]
    assert len(threshold_texts) == 2
    for t in threshold_texts:
        # x in data coords, y in axes-fraction coords (ax.get_xaxis_transform()), not a data-ylim-derived y.
        assert t.get_transform() == ax.get_xaxis_transform()


def test_bench_without_ext_shows_p3_skipped(tmp_path, monkeypatch):
    df = synthetic_dataset(3, with_ext=False)
    out = _run_bench_on(df, tmp_path, monkeypatch)
    text = out.read_text()
    assert "Not run: TalkingHeadBench could not be fetched (P3 skipped)." in text


def test_bench_omits_ibeta_footnote_when_no_cell_is_small(tmp_path, monkeypatch):
    """The "* fewer than 150 attempts" footnote must only print when at least one rendered iBeta cell is
    actually marked with the asterisk (n < config.N_ATTACK)."""
    df = synthetic_dataset(3, with_ext=True)  # 36 videos per attack type
    monkeypatch.setattr(bench.config, "N_ATTACK", 10)  # every attack type now has n (36) >= N_ATTACK
    out = _run_bench_on(df, tmp_path, monkeypatch)
    text = out.read_text()
    assert "fewer than 150 attempts" not in text
    assert "*" not in text.split("## 6. iBeta-style")[1].split("## 7.")[0]


def test_group_table_auc_rendered_as_number_not_percentage(tmp_path, monkeypatch):
    """§4's per-held-group table renders auc as a plain 3-decimal number, like p1_tbl/p2_tbl, not as a
    percentage."""
    df = synthetic_dataset(3, with_ext=True)
    out = _run_bench_on(df, tmp_path, monkeypatch)
    text = out.read_text()
    section = text.split("Per held-out group, risk engine")[1].split("Per attack type")[0]
    header, _, *rows = [ln for ln in section.splitlines() if ln.strip().startswith("|")]
    auc_idx = [c.strip() for c in header.strip("|").split("|")].index("auc")
    for row in rows:
        cell = row.strip("|").split("|")[auc_idx].strip()
        assert cell == "—" or re.fullmatch(r"\d+\.\d{3}", cell), cell


def test_load_dataset_raises_on_missing_pad_sbi_columns(tmp_path, monkeypatch):
    """A broken/partial extract run that dropped the pad_* columns entirely kills the MiniFASNet baseline:
    load_dataset must now raise SystemExit (the comparison would be meaningless) instead of silently
    reindexing them in as NaN."""
    import pytest

    df = synthetic_dataset(2)
    manifest = df[["vid", "label", "attack", "group", "actor", "fold"]].copy()
    features = df.drop(columns=["label", "attack", "group", "actor", "fold",
                                "pad_real_mean", "pad_real_min", "pad_real_low4", "pad_real_std"]).copy()
    features["no_face"] = False
    manifest_csv, features_csv = tmp_path / "manifest.csv", tmp_path / "features.csv"
    manifest.to_csv(manifest_csv, index=False)
    features.to_csv(features_csv, index=False)
    monkeypatch.setattr(bench.config, "MANIFEST", manifest_csv)
    monkeypatch.setattr(bench.config, "FEATURES", features_csv)

    with pytest.raises(SystemExit, match="pad_real_mean"):
        bench.load_dataset()


def test_bench_aborts_on_missing_pad_columns(tmp_path, monkeypatch):
    """Full pipeline: run_bench must abort (not silently produce a bogus report) when the features file is
    missing the pad_* columns -- the MiniFASNet baseline comparison would be meaningless."""
    import pytest

    df = synthetic_dataset(3, with_ext=True)
    manifest = df[["vid", "label", "attack", "group", "actor", "fold"]].copy()
    features = df.drop(columns=["label", "attack", "group", "actor", "fold",
                                "pad_real_mean", "pad_real_min", "pad_real_low4", "pad_real_std"]).copy()
    features["no_face"] = False
    manifest_csv, features_csv = tmp_path / "manifest.csv", tmp_path / "features.csv"
    manifest.to_csv(manifest_csv, index=False)
    features.to_csv(features_csv, index=False)
    monkeypatch.setattr(bench.config, "MANIFEST", manifest_csv)
    monkeypatch.setattr(bench.config, "FEATURES", features_csv)
    monkeypatch.setattr(bench.config, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(bench.config, "FIGURES", tmp_path / "figures")
    monkeypatch.setattr(bench.config, "REPORTS", tmp_path)
    monkeypatch.setattr(bench.config, "BUNDLE", tmp_path / "bundle.joblib")
    monkeypatch.setattr(bench, "ABLATION_GROUPS", {"pad": "pad_", "blend": "bn_"})

    with pytest.raises(SystemExit):
        bench.run_bench()


def test_load_dataset_raises_when_baseline_mostly_nan(tmp_path, monkeypatch):
    """Columns present but mostly NaN (> 20% of non-EXT rows) is just as dead as missing entirely."""
    import pytest

    df = synthetic_dataset(2)
    manifest = df[["vid", "label", "attack", "group", "actor", "fold"]].copy()
    features = df.drop(columns=["label", "attack", "group", "actor", "fold"]).copy()
    features["no_face"] = False
    features.loc[features.index[: len(features) // 2], "sbi_mean"] = float("nan")
    manifest_csv, features_csv = tmp_path / "manifest.csv", tmp_path / "features.csv"
    manifest.to_csv(manifest_csv, index=False)
    features.to_csv(features_csv, index=False)
    monkeypatch.setattr(bench.config, "MANIFEST", manifest_csv)
    monkeypatch.setattr(bench.config, "FEATURES", features_csv)

    with pytest.raises(SystemExit, match="sbi_mean"):
        bench.load_dataset()


def test_load_dataset_ignores_no_face_rows_for_baseline_nan_check(tmp_path, monkeypatch):
    """A baseline is expected to be NaN on no_face rows (no face to score); those rows must not count toward
    the 20% dead-baseline threshold."""
    df = synthetic_dataset(2)
    manifest = df[["vid", "label", "attack", "group", "actor", "fold"]].copy()
    features = df.drop(columns=["label", "attack", "group", "actor", "fold"]).copy()
    features["no_face"] = False
    # Mark every bona fide row no_face and NaN out sbi_mean there: with per_actor=2 this is well over 20% of
    # all rows, but 0% of the non-EXT, non-no_face rows, so load_dataset must not raise.
    bona_mask = manifest.label == 0
    features.loc[bona_mask.to_numpy(), "no_face"] = True
    features.loc[bona_mask.to_numpy(), "sbi_mean"] = float("nan")
    manifest_csv, features_csv = tmp_path / "manifest.csv", tmp_path / "features.csv"
    manifest.to_csv(manifest_csv, index=False)
    features.to_csv(features_csv, index=False)
    monkeypatch.setattr(bench.config, "MANIFEST", manifest_csv)
    monkeypatch.setattr(bench.config, "FEATURES", features_csv)

    out_df, dropped, n_no_feat = bench.load_dataset()
    assert len(dropped) == int(bona_mask.sum())
    assert not out_df["sbi_mean"].isna().any()


def test_review_load_uses_each_rows_own_thresholds():
    # Two "split models" with very different thresholds: a row must be judged against its own split's
    # tau_review/tau_deny, not a pooled value -- pooling would misroute split B's attack here.
    res = pd.DataFrame({
        "score":      [0.5, 0.5, 0.0],
        "label":      [1,   1,   0],
        "tau_review": [0.1, 0.9, 0.1],
        "tau_deny":   [0.2, 0.95, 0.2],
    })
    out = recommend.review_load(res, prevalences=(1.0,))
    # split A's row (score 0.5 >= tau_deny 0.2) is NOT in the review band; split B's row (score 0.5 < tau_review
    # 0.9) is also not in the band -- so with attack prevalence 1.0, 0% are routed to review by the band.
    assert out.iloc[0]["sessions routed to review by the risk band"] == "0.0%"


def test_routing_table_matches_policy_decide_order():
    src = Path(policy_mod.__file__).read_text()
    expected = re.findall(r'Decision\("(\w+)",\s*"(\w+)"\)', src)
    assert expected, "could not find any Decision(...) literals in policy.py"

    table = recommend._routing_table()
    # Only the trailing "reason code" column, not any backtick-quoted identifier inside the condition text.
    found = re.findall(r"`(\w+)`\s*\|\s*$", table, re.MULTILINE)
    assert found == [reason for _, reason in expected]
