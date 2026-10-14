import numpy as np

from fdemo.eval.metrics import auc
from fdemo.fusion.model import CONFIGS, Scorer, _groups, config_columns
from synth import synthetic_dataset


def test_config_columns():
    df = synthetic_dataset(2)
    assert config_columns("B1", df) == ["pad_real_mean"]
    assert config_columns("B2", df) == ["sbi_mean"]
    f = config_columns("F", df)
    assert "bn_core_bg" in f and "pad_real_mean" not in f and "noise_col" not in f
    assert set(config_columns("ALL", df)) >= set(f) | {"pad_real_mean", "sbi_mean"}


def test_all_configs_fit_and_separate():
    df = synthetic_dataset(5)
    tr, te = df[df.fold != 0], df[df.fold == 0]
    for cfg in CONFIGS:
        s = Scorer(cfg).fit(tr)
        p = s.predict(te)
        assert len(p) == len(te) and np.isfinite(p).all()
        assert auc(p[te.label.to_numpy() == 0], p[te.label.to_numpy() == 1]) > 0.7, cfg
        assert len(s.dev_scores(tr)) == len(tr)


def test_raw_baseline_fills_nan():
    df = synthetic_dataset(2)
    df.loc[df.index[:3], "pad_real_mean"] = np.nan
    s = Scorer("B1").fit(df)
    assert np.isfinite(s.predict(df)).all()


def test_fusion_handles_missing_values():
    df = synthetic_dataset(3)
    df.loc[df.index[::7], "rp_snr"] = np.nan
    s = Scorer("ALL").fit(df)
    assert np.isfinite(s.predict(df)).all() and s.kind in ("lr", "hgb")


def test_inner_cv_groups_by_outer_fold_not_actor():
    # Inner CV must group by the outer fold (keeps a face-swap's source/target identities together),
    # not by actor.
    df = synthetic_dataset(2)
    assert np.array_equal(_groups(df), df["fold"].to_numpy())
    assert not np.array_equal(_groups(df), df["actor"].to_numpy())
