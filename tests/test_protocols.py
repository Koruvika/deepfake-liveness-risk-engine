import numpy as np
import pandas as pd

from fdemo.eval.protocols import p1_splits, p2_splits, p3_splits, run_protocol
from fdemo.eval.summary import attack_table, overall, success_check, summarize_p2, summarize_p3
from synth import synthetic_dataset


def test_p1_tests_every_synthetic_sample_once():
    df = synthetic_dataset(2)
    hits = np.sum([s.test for s in p1_splits(df)], axis=0)
    assert (hits == 1).all()


def test_p2_has_no_identity_or_attack_leakage():
    df = synthetic_dataset(2)
    splits = p2_splits(df)
    assert len(splits) == 16
    for s in splits:
        tr, te = df[s.train], df[s.test]
        assert not set(tr.actor) & set(te.actor)
        held = s.name.split("/")[0]
        assert held not in set(tr.group)
        assert set(te.group) <= {"bona", held}
        if held == "G2":
            assert not {"A2", "A4"} & set(tr.attack) and {"A2", "A4"} <= set(te.attack)


def test_p3_empty_without_external():
    assert p3_splits(synthetic_dataset(2)) == []


def test_run_protocol_and_summaries():
    df = synthetic_dataset(3, with_ext=True)
    res = run_protocol(df, "ALL", p2_splits(df))
    assert {"score", "tau_op", "tau_review", "tau_deny", "split"} <= set(res.columns)
    per_group, per_attack, macro = summarize_p2(res)
    assert len(per_group) == 4 and set(per_attack.attack) == {"A1", "A2", "A3", "A4", "A5"}
    assert 0 <= macro["acer"] <= 1
    assert {"group_boot_lo", "group_boot_hi", "group_apcer"} <= set(per_group.columns)
    assert {"boot_lo", "boot_hi"} <= set(per_attack.columns)
    p1 = run_protocol(df, "B1", p1_splits(df))
    o = overall(p1)
    assert {"bpcer", "apcer_max", "acer", "auc"} <= set(o)
    tbl = attack_table(p1)
    assert {"apcer", "lo", "hi", "boot_lo", "boot_hi", "ibeta_L1", "ibeta_L2"} <= set(tbl.columns)
    p3 = summarize_p3(run_protocol(df, "ALL", p3_splits(df)))
    assert list(p3.attack) == ["EXT_Hallo"] and p3.n.iloc[0] == 10


def test_overall_auc_is_scale_invariant_across_splits():
    # AUC is invariant to a monotone per-split rescaling; per-split-then-average must equal the
    # single-split AUC, not a pooled-score AUC that would be thrown off by inconsistent scales.
    rng = np.random.default_rng(0)
    bona_a = rng.uniform(0, 0.5, 50)
    att_a = rng.uniform(0.3, 1.0, 50)
    single = pd.DataFrame({
        "label": [0] * 50 + [1] * 50, "score": np.r_[bona_a, att_a], "attack": "bona",
        "split": "fold0", "actor": np.arange(100), "tau_op": 0.5,
    })
    bona_b, att_b = bona_a * 10 + 5, att_a * 10 + 5
    two_split = pd.concat([
        single,
        pd.DataFrame({"label": [0] * 50 + [1] * 50, "score": np.r_[bona_b, att_b], "attack": "bona",
                      "split": "fold1", "actor": np.arange(100), "tau_op": 5.5}),
    ], ignore_index=True)
    assert overall(two_split)["auc"] == overall(single)["auc"]


def test_summarize_p3_majority_vote_ties_count_as_accepted():
    res = pd.DataFrame({
        "vid": ["e1", "e1", "e2", "e2"], "label": [1, 1, 1, 1], "attack": ["EXT_X"] * 4,
        "group": ["EXT"] * 4, "actor": [-1] * 4, "score": [0.1, 0.9, 0.9, 0.9], "tau_op": 0.5,
        "tau_review": 0.3, "tau_deny": 0.7,
    })
    p3 = summarize_p3(res)
    row = p3[p3.attack == "EXT_X"].iloc[0]
    # e1: accepted in 1/2 splits (tie) -> counts as accepted. e2: accepted in 0/2 -> not accepted.
    assert row.n == 2 and row.apcer == 0.5


def test_group_boot_matches_attack_boot_for_single_attack_group():
    res = pd.DataFrame({
        "vid": [f"v{i}" for i in range(40)],
        "label": [0] * 20 + [1] * 20,
        "attack": ["bona"] * 20 + ["A1"] * 20,
        "group": ["bona"] * 20 + ["G1"] * 20,
        "actor": list(range(10)) * 2 + list(range(10)) * 2,
        "score": np.r_[np.linspace(0, 0.4, 20), np.linspace(0.3, 0.9, 20)],
        "split": "G1/fold0",
        "tau_op": 0.5, "tau_review": 0.3, "tau_deny": 0.7,
    })
    per_group, per_attack, _ = summarize_p2(res)
    g = per_group[per_group.held == "G1"].iloc[0]
    a = per_attack[(per_attack.held == "G1") & (per_attack.attack == "A1")].iloc[0]
    assert abs(g.group_boot_lo - a.boot_lo) < 1e-9
    assert abs(g.group_boot_hi - a.boot_hi) < 1e-9


def test_group_boot_ci_contains_point_estimate_for_two_attack_group():
    df = synthetic_dataset(4)
    res = run_protocol(df, "ALL", p2_splits(df))
    per_group, _, _ = summarize_p2(res)
    g2 = per_group[per_group.held == "G2"].iloc[0]
    assert g2.group_boot_lo <= g2.group_apcer <= g2.group_boot_hi


def test_summarize_p2_drops_unused_legacy_ci_columns():
    df = synthetic_dataset(2)
    per_group, _, _ = summarize_p2(run_protocol(df, "ALL", p2_splits(df)))
    assert "group_lo" not in per_group.columns and "group_hi" not in per_group.columns


def test_success_check_separated_vs_overlapping():
    base = pd.DataFrame({"held": ["G1", "G2", "G3", "G4"], "group_boot_lo": [0.5, 0.5, 0.5, 0.5],
                         "group_boot_hi": [0.6, 0.6, 0.6, 0.6]})
    separated = pd.DataFrame({"held": ["G1", "G2", "G3", "G4"], "group_boot_lo": [0.01, 0.01, 0.01, 0.01],
                              "group_boot_hi": [0.05, 0.05, 0.05, 0.05]})
    overlapping = pd.DataFrame({"held": ["G1", "G2", "G3", "G4"], "group_boot_lo": [0.4, 0.4, 0.4, 0.4],
                                "group_boot_hi": [0.55, 0.55, 0.55, 0.55]})
    macro_better = {"acer": 0.05}
    macro_worse = {"acer": 0.2}
    out_pass = success_check(separated, macro_better, base, macro_worse)
    assert out_pass["passed"] and out_pass["acer_better"] and out_pass["groups_nonoverlap"] == 4
    out_fail = success_check(overlapping, macro_better, base, macro_worse)
    assert not out_fail["passed"] and out_fail["groups_nonoverlap"] == 0


def test_success_check():
    df = synthetic_dataset(3)
    g_all, _, m_all = summarize_p2(run_protocol(df, "ALL", p2_splits(df)))
    g_b1, _, m_b1 = summarize_p2(run_protocol(df, "B1", p2_splits(df)))
    out = success_check(g_all, m_all, g_b1, m_b1)
    assert {"passed", "acer_better", "groups_nonoverlap"} <= set(out)
