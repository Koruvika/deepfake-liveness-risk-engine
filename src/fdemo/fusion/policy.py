"""Two-threshold routing (allow / review / deny) combined with the vendor (FaceTec) result."""
from dataclasses import dataclass

import numpy as np

from fdemo.eval.metrics import threshold_at_apcer, threshold_at_bpcer

DENY_BPCER = 0.01       # at most 1% of genuine users auto-denied
ALLOW_APCER = 0.05      # at most 5% of attacks auto-allowed
MIN_MATCH_LEVEL = 6     # FaceTec 3D:2D level 6 = 1/100,000 FAR (default 4 = 1/1,000)


@dataclass(frozen=True)
class Thresholds:
    review: float
    deny: float

    @property
    def has_review_band(self) -> bool:
        return self.review < self.deny


@dataclass(frozen=True)
class Decision:
    action: str   # "allow" | "review" | "deny"
    reason: str


def make_thresholds(dev_bona, dev_attack, deny_bpcer=DENY_BPCER, allow_apcer=ALLOW_APCER) -> Thresholds:
    """Fail closed: non-finite dev scores are dropped before threshold-picking, and if that leaves either
    side empty we raise rather than silently returning a degenerate (e.g. all -inf/inf) threshold."""
    bona = np.asarray(dev_bona, float)
    bona = bona[np.isfinite(bona)]
    attack = np.asarray(dev_attack, float)
    attack = attack[np.isfinite(attack)]
    if len(bona) == 0 or len(attack) == 0:
        raise ValueError("make_thresholds requires at least one finite bona fide and one finite attack dev score")
    deny = threshold_at_bpcer(bona, deny_bpcer)
    review = threshold_at_apcer(attack, allow_apcer)
    # review > deny only when the classes are separated: one threshold meets both targets, no review band
    return Thresholds(review=min(review, deny), deny=deny)


def decide(risk, th, vendor=None, photo_id=False, no_face=False, signal_failure=False) -> Decision:
    if vendor is not None and not vendor.liveness_proven:
        return Decision("deny", "vendor_liveness_not_proven")
    if np.isnan(th.review) or np.isnan(th.deny):
        return Decision("review", "thresholds_invalid")
    if no_face:
        # no_face must not be denied on imputed/fabricated risk features, regardless of the risk score.
        return Decision("review", "no_face")
    if signal_failure:
        # a signal that errored out gets median-imputed by the model's imputer; the resulting risk score is
        # fabricated, not measured, so it must not be trusted to allow or deny.
        return Decision("review", "signal_failure")
    if not np.isfinite(risk):
        return Decision("review", "risk_unavailable")
    if risk >= th.deny:
        return Decision("deny", "risk_high")
    if risk >= th.review:
        return Decision("review", "risk_medium")
    if photo_id and (vendor is None or vendor.match_level is None or vendor.match_level < MIN_MATCH_LEVEL):
        return Decision("review", "weak_photo_id_match")
    return Decision("allow", "ok")
