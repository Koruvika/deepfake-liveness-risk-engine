import numpy as np
import pytest

from fdemo.fusion.policy import Thresholds, decide, make_thresholds
from fdemo.fusion.vendor_adapter import VendorResult

TH = Thresholds(review=0.3, deny=0.7)


def v(proven=True, level=None):
    return VendorResult(proven, level, None, "u1", "v10")


@pytest.mark.parametrize("vendor,risk,photo,expected", [
    (v(False), 0.0, False, ("deny", "vendor_liveness_not_proven")),
    (v(), 0.8, False, ("deny", "risk_high")),
    (v(), 0.5, False, ("review", "risk_medium")),
    (v(level=5), 0.1, True, ("review", "weak_photo_id_match")),
    (v(level=6), 0.1, True, ("allow", "ok")),
    (v(), 0.1, False, ("allow", "ok")),
    (None, 0.1, False, ("allow", "ok")),
    (v(), float("nan"), False, ("review", "risk_unavailable")),
])
def test_decision_table(vendor, risk, photo, expected):
    d = decide(risk, TH, vendor=vendor, photo_id=photo)
    assert (d.action, d.reason) == expected


def test_no_face_goes_to_review():
    assert decide(0.1, TH, vendor=v(), no_face=True).action == "review"


def test_signal_failure_goes_to_review():
    d = decide(0.9, TH, vendor=v(), signal_failure=True)
    assert (d.action, d.reason) == ("review", "signal_failure")


def test_signal_failure_overrides_high_risk_but_not_no_face():
    # no_face must win over signal_failure since it is checked first.
    d = decide(0.9, TH, vendor=v(), no_face=True, signal_failure=True)
    assert (d.action, d.reason) == ("review", "no_face")


def test_make_thresholds_targets():
    bona = np.linspace(0, 0.5, 200)
    att = np.linspace(0.3, 1.0, 200)
    th = make_thresholds(bona, att)
    assert np.mean(bona >= th.deny) <= 0.01
    assert np.mean(att < th.review) <= 0.05
    assert th.review <= th.deny


def test_make_thresholds_single_threshold_when_separated():
    # attack 5th percentile above bona 99th percentile: one threshold meets both targets, no review band
    bona = np.linspace(0, 0.4, 200)
    att = np.linspace(0.6, 1.0, 200)
    th = make_thresholds(bona, att)
    assert th.review == th.deny and not th.has_review_band
    assert np.mean(bona >= th.deny) <= 0.01 and np.mean(att < th.review) <= 0.05


def test_make_thresholds_drops_non_finite_scores():
    bona = np.r_[np.linspace(0, 0.5, 190), np.full(10, np.nan)]
    att = np.r_[np.linspace(0.3, 1.0, 190), np.full(10, np.inf)]
    th = make_thresholds(bona, att)
    assert np.isfinite(th.review) and np.isfinite(th.deny)


def test_make_thresholds_raises_when_bona_all_nan():
    with pytest.raises(ValueError):
        make_thresholds(np.full(10, np.nan), np.linspace(0, 1, 10))


def test_make_thresholds_raises_when_attack_all_nan():
    with pytest.raises(ValueError):
        make_thresholds(np.linspace(0, 1, 10), np.full(10, np.nan))


def test_thresholds_invalid_goes_to_review():
    th = Thresholds(float("nan"), float("nan"))
    d = decide(0.1, th, vendor=v())
    assert (d.action, d.reason) == ("review", "thresholds_invalid")


def test_no_face_overrides_high_risk():
    d = decide(0.99, TH, vendor=v(), no_face=True)
    assert (d.action, d.reason) == ("review", "no_face")
