import numpy as np
import pytest

from fdemo.eval.metrics import (
    acer, apcer, apcer_at_bpcer, auc, bpcer, bpcer_at_apcer, clopper_pearson, cluster_bootstrap_ci,
    ibeta_indicative, rate_with_ci, threshold_at_apcer, threshold_at_bpcer,
)


def test_apcer_bpcer_convention():
    assert apcer([0.1, 0.9, 0.4], 0.5) == pytest.approx(2 / 3)
    assert bpcer([0.1, 0.9, 0.5], 0.5) == pytest.approx(2 / 3)


def test_threshold_at_bpcer():
    bona = np.arange(100) / 100
    assert bpcer(bona, threshold_at_bpcer(bona, 0.01)) <= 0.01
    assert bpcer(bona, threshold_at_bpcer(bona, 0.05)) == pytest.approx(0.05)


def test_threshold_at_apcer():
    att = np.arange(100) / 100
    assert apcer(att, threshold_at_apcer(att, 0.05)) == pytest.approx(0.05)


def test_curve_points():
    bona = np.linspace(0, 0.5, 100)
    att = np.linspace(0.4, 1.0, 100)
    assert bpcer_at_apcer(bona, att, 0.05) < 0.2
    assert apcer_at_bpcer(bona, att, 0.01) < 0.2
    assert auc(bona, att) > 0.9


def test_clopper_pearson_zero_errors_matches_rule_of_three():
    lo, hi = clopper_pearson(0, 150)
    assert lo == 0.0 and 0.02 < hi < 0.026


def test_rate_with_ci():
    r = rate_with_ci(np.array([True, False, False, False]))
    assert (r["k"], r["n"]) == (1, 4) and r["rate"] == 0.25 and r["lo"] < 0.25 < r["hi"]


@pytest.mark.parametrize("k,n,b,l1,l2", [
    (0, 150, 0.10, True, True),
    (1, 150, 0.10, False, True),
    (2, 150, 0.10, False, False),
    (0, 150, 0.20, False, False),
])
def test_ibeta_indicative(k, n, b, l1, l2):
    assert ibeta_indicative(k, n, b) == {"L1": l1, "L2": l2}


def test_acer():
    assert acer(0.2, 0.1) == pytest.approx(0.15)


def test_ibeta_indicative_zero_trials_is_not_a_pass():
    assert ibeta_indicative(0, 0, 0.1) == {"L1": False, "L2": False}


def test_auc_empty_input_is_nan():
    assert np.isnan(auc([], [0.5]))


def test_cluster_bootstrap_ci_empty_is_nan():
    lo, hi = cluster_bootstrap_ci(np.array([], bool), np.array([]))
    assert np.isnan(lo) and np.isnan(hi)


def test_cluster_bootstrap_ci_zero_errors_envelope_not_narrower_than_clopper_pearson():
    # 0 errors in all 12 clusters: the percentile bootstrap alone collapses to [0, 0], which is narrower
    # than the exact interval and would wrongly inflate iBeta L1 confidence -- envelope must widen it back.
    clusters = np.repeat(np.arange(12), 25)
    errors = np.zeros(300, bool)
    cp_lo, cp_hi = clopper_pearson(0, 300)
    boot_lo, boot_hi = cluster_bootstrap_ci(errors, clusters, n_boot=500, seed=0)
    assert boot_hi > 0
    assert boot_hi >= cp_hi
    assert boot_lo <= cp_lo


def test_cluster_bootstrap_ci_one_error_envelope_not_narrower_than_clopper_pearson():
    clusters = np.repeat(np.arange(12), 25)
    errors = np.zeros(300, bool)
    errors[0] = True
    cp_lo, cp_hi = clopper_pearson(1, 300)
    boot_lo, boot_hi = cluster_bootstrap_ci(errors, clusters, n_boot=500, seed=0)
    assert boot_lo <= cp_lo
    assert boot_hi >= cp_hi


def test_cluster_bootstrap_ci_wider_than_clopper_pearson_when_clustered():
    # All 30 errors concentrated in 1 of 6 clusters: clustering means the effective sample size is much
    # smaller than 180*6, so the cluster bootstrap CI should be at least as wide as plain Clopper-Pearson.
    clusters = np.repeat(np.arange(6), 30)
    errors = np.zeros(180, bool)
    errors[:30] = True  # all errors in cluster 0
    cp_lo, cp_hi = clopper_pearson(int(errors.sum()), len(errors))
    boot_lo, boot_hi = cluster_bootstrap_ci(errors, clusters, n_boot=1000, seed=0)
    point = errors.mean()
    assert boot_lo <= point <= boot_hi
    assert (boot_hi - boot_lo) >= (cp_hi - cp_lo)
