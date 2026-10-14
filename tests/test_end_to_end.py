import pandas as pd

from fdemo.eval.end_to_end import SCENARIOS, bpcer_breakdown, e2e_table


def toy():
    res = pd.DataFrame({
        "vid": ["b1", "b2", "a1", "a2"], "label": [0, 0, 1, 1], "attack": ["bona", "bona", "A2", "A2"],
        "score": [0.1, 0.9, 0.9, 0.1], "tau_review": 0.5, "tau_deny": 0.8,
    })
    feats = pd.DataFrame({"vid": ["b1", "b2", "a1", "a2"], "pad_real_mean": [0.9, 0.2, 0.9, 0.9]})
    return res, feats


def test_bypassed_vendor_only_allows_everything():
    t = e2e_table(*toy())
    row = t[(t.scenario == "V-bypassed") & (t["mode"] == "vendor-only") & (t.attack == "A2")].iloc[0]
    assert row.allow == 1.0 and row.deny == 0.0


def test_engine_catches_attack_when_vendor_bypassed():
    t = e2e_table(*toy())
    row = t[(t.scenario == "V-bypassed") & (t["mode"] == "vendor+engine") & (t.attack == "A2")].iloc[0]
    assert row.deny == 0.5 and row.allow == 0.5


def test_standin_denies_low_pad_bona():
    t = e2e_table(*toy())
    row = t[(t.scenario == "V-standin") & (t["mode"] == "vendor-only") & (t.attack == "bona")].iloc[0]
    assert row.deny == 0.5
    assert set(t.scenario) == set(SCENARIOS)


def test_bpcer_breakdown_attributes_denials():
    b = bpcer_breakdown(*toy()).set_index("scenario")
    assert b.loc["V-standin", "deny_by_vendor"] == 0.5
    assert b.loc["V-ideal-bonafide", "deny_by_vendor"] == 0.0
    assert b.loc["V-ideal-bonafide", "deny_by_engine"] == 0.5


def test_e2e_table_has_n_videos():
    t = e2e_table(*toy())
    row = t[(t.scenario == "V-bypassed") & (t["mode"] == "vendor-only") & (t.attack == "A2")].iloc[0]
    assert row.n_videos == 2


def test_e2e_table_err_row_is_reviewed_not_denied():
    res, feats = toy()
    # a1 (attack, score 0.9) would otherwise be denied under V-bypassed/vendor+engine; a failing signal must
    # route it to review instead, pulling the attack-level deny rate below 1.0.
    feats = feats.assign(err_sbi=[0, 0, 1, 0])
    t = e2e_table(res, feats, nan_free_cols=[])
    row = t[(t.scenario == "V-bypassed") & (t["mode"] == "vendor+engine") & (t.attack == "A2")].iloc[0]
    assert row.deny == 0.0 and row.review == 0.5 and row.allow == 0.5  # a1 -> review, a2 -> allow (score 0.1)


def test_bpcer_breakdown_has_review_signal_failure_column():
    res, feats = toy()
    feats = feats.assign(err_sbi=[1, 0, 0, 0])  # b1 (bona, label 0) has a failing signal
    b = bpcer_breakdown(res, feats, nan_free_cols=[]).set_index("scenario")
    # b1 would otherwise allow (score 0.1 < tau_review 0.5); the signal failure must override that to review.
    assert b.loc["V-bypassed", "review_signal_failure"] == 0.5
    assert "review_signal_failure" in b.columns


def test_bpcer_breakdown_nan_free_col_nan_routes_to_review_signal_failure():
    res, feats = toy()
    feats = feats.assign(pad_real_std=[float("nan"), 0.1, 0.1, 0.1])  # b1's nan_free col is NaN
    b = bpcer_breakdown(res, feats, nan_free_cols=["pad_real_std"]).set_index("scenario")
    assert b.loc["V-bypassed", "review_signal_failure"] == 0.5
