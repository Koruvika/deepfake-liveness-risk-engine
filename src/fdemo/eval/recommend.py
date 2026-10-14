"""Deliverable 4: thresholds, routing, step-up, monitoring and manual-review rules."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fdemo import config
from fdemo.eval.report import md_table, pct
from fdemo.fusion.policy import ALLOW_APCER, DENY_BPCER, MIN_MATCH_LEVEL

# Routing rules in the exact order implemented by `fdemo.fusion.policy.decide`. Keep this in sync with that
# function: tests/test_report.py regex-extracts the Decision(...) literals from policy.py source and checks
# this table lists the same reason codes in the same order.
ROUTING_RULES = [
    ("FaceTec liveness not proven **in this session** (persist `livenessProven` from the liveness step)",
     "deny, allow retry, count velocity", "vendor_liveness_not_proven"),
    ("thresholds could not be computed (dev set had no finite score for one class)",
     "review", "thresholds_invalid"),
    ("no face in >50% of frames (risk features are imputed, not measured)", "review", "no_face"),
    ("a signal errored out, or a trained feature is missing from the row (imputed, not measured)",
     "review", "signal_failure"),
    ("risk score is not finite", "review", "risk_unavailable"),
    ("risk ≥ τ_deny", "deny, log, flag to fraud team", "risk_high"),
    ("τ_review ≤ risk < τ_deny", "manual review", "risk_medium"),
    ("Photo ID flow and `matchLevel` < {min_match_level}", "review", "weak_photo_id_match"),
    ("otherwise", "allow", "ok"),
]


def psi(expected, actual, bins=10) -> float:
    """Population Stability Index of the risk-score distribution (monitoring drift)."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def sweep(res, taus) -> pd.DataFrame:
    bona, att = res[res.label == 0].score.to_numpy(), res[res.label == 1].score.to_numpy()
    return pd.DataFrame({"tau": taus,
                         "APCER (attacks allowed)": [np.mean(att < t) for t in taus],
                         "BPCER (genuine flagged)": [np.mean(bona >= t) for t in taus]})


def review_load(res, prevalences=(0.005, 0.02, 0.10)) -> pd.DataFrame:
    """Each row is routed against its own split's tau_review/tau_deny (as eval.end_to_end does), not a single
    pooled threshold -- the 16 split models were trained on different folds and are not directly comparable on
    a common score scale."""
    s, y = res.score.to_numpy(), res.label.to_numpy()
    tr, td = res.tau_review.to_numpy(), res.tau_deny.to_numpy()
    band = (s >= tr) & (s < td)
    br, ar = band[y == 0].mean(), band[y == 1].mean()
    return pd.DataFrame([{"attack prevalence": pct(p),
                          "sessions routed to review by the risk band": pct(p * ar + (1 - p) * br)}
                         for p in prevalences])


def _deployed_thresholds() -> dict | None:
    p = config.RESULTS / "final_thresholds.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _routing_table() -> str:
    rows = pd.DataFrame([{"order": i + 1, "condition": cond.format(min_match_level=MIN_MATCH_LEVEL),
                          "action": action, "reason code": f"`{reason}`"}
                         for i, (cond, action, reason) in enumerate(ROUTING_RULES)])
    return md_table(rows)


def write_recommendations(results_csv=None, out=None) -> Path:
    results_csv = results_csv or config.RESULTS / "P2_ALL.csv"
    out = out or config.REPORTS / "recommendations.md"
    try:
        res = pd.read_csv(results_csv)
    except FileNotFoundError:
        raise SystemExit(f"results CSV not found at {results_csv}; run `fdemo bench` first")
    per_split = res.groupby("split")[["tau_review", "tau_deny"]].first()
    tau_r, tau_d = float(per_split.tau_review.median()), float(per_split.tau_deny.median())
    deployed = _deployed_thresholds()
    deployed_row = (f"| τ_review | {deployed['review']:.3f} | {tau_r:.3f} | at most {pct(ALLOW_APCER)} of attacks "
                    f"below it (auto-allowed) |\n"
                    f"| τ_deny | {deployed['deny']:.3f} | {tau_d:.3f} | at most {pct(DENY_BPCER)} of genuine users "
                    f"above it (auto-denied) |") if deployed else (
                    f"| τ_review | — (bundle not trained yet) | {tau_r:.3f} | at most {pct(ALLOW_APCER)} of attacks "
                    f"below it (auto-allowed) |\n"
                    f"| τ_deny | — (bundle not trained yet) | {tau_d:.3f} | at most {pct(DENY_BPCER)} of genuine "
                    f"users above it (auto-denied) |")
    taus = np.round(np.linspace(0.05, 0.95, 19), 2)
    sw = sweep(res, taus)
    bona = res[res.label == 0]
    att = res[res.label == 1]
    # Per-row application of each split model's own tau_review/tau_deny (as eval.end_to_end does) -- the 16
    # split models were trained on different folds and are not directly comparable on one pooled score scale.
    att_allowed = (att.score.to_numpy() < att.tau_review.to_numpy()).mean()
    bona_denied = (bona.score.to_numpy() >= bona.tau_deny.to_numpy()).mean()
    bona_reviewed = ((bona.score.to_numpy() >= bona.tau_review.to_numpy())
                     & (bona.score.to_numpy() < bona.tau_deny.to_numpy())).mean()
    text = f"""# Recommendations: thresholds, routing, monitoring and manual review

Source: `{Path(results_csv).name}` — risk engine (ALL) on protocol P2 (attack group never seen in training,
identity-disjoint folds). Numbers come from simulated attacks; re-run this sweep on the client's own
labelled sessions before go-live.

## Threshold sweep (pooled across split models — indicative only)

{md_table(sw, {"tau": lambda v: f"{v:.2f}", "APCER (attacks allowed)": pct, "BPCER (genuine flagged)": pct})}

## Recommended operating point

| threshold | deployed in `fdemo scan` (fitted on all synthetic data) | validated on P2 (median of per-split thresholds) | rule used to pick it on dev data |
|---|---|---|---|
{deployed_row}

At this point on P2 (validated thresholds, each session judged against its own split model's thresholds):
attacks auto-allowed {pct(att_allowed)}, genuine users auto-denied {pct(bona_denied)}, genuine users sent to
review {pct(bona_reviewed)}.
{"" if tau_r < tau_d else "**Scores are separated on this data: a single threshold (τ_deny) meets both targets, so there is no review band.**"}

### Expected manual-review load

{md_table(review_load(res))}

## Routing rules (implemented in `fdemo.fusion.policy.decide`, in this exact order)

{_routing_table()}

## FaceTec configuration

- Set `minimumMatchLevel3D2DMatching` to **≥ {MIN_MATCH_LEVEL}** (1/100,000 FAR). The default 4 (1/1,000 FAR) is not a strong match
  according to FaceTec's own guidance; levels 4–5 should route to review, never auto-approve.
- When the document has a chip, require `matchLevelNFCToFaceMap` ≥ {MIN_MATCH_LEVEL} and prefer it over the printed-photo match.
- Verify every session **server-side** (backend → FaceTec Server); never accept a result flag sent by the client app.
- Bind each session to a server-issued nonce and the `externalDatabaseRefID`; reject results whose session/reference
  does not match the user and transaction being authorised.
- 3D:3D re-verification (fixed level 15) for high-risk actions: payout, device/phone/e-mail change.

## Step-up and fallback

- Step-up ladder for `review` and repeated borderline sessions: NFC chip read → 3D:3D re-verification on a different
  device → scripted live video call with a trained agent.
- **Never fall back to a weaker channel** (selfie upload, unscripted support call, e-mail reset) when liveness fails.
- After 3 failed sessions in 24 h for the same user, device fingerprint or IP: 24 h cooldown, then step-up only.

## Monitoring

| metric | granularity | alert when |
|---|---|---|
| allow / review / deny rates | platform × app version × day | any rate moves > 3 percentage points vs the trailing 4-week mean |
| risk-score drift (PSI, `fdemo.eval.recommend.psi`) | week | PSI > 0.2 (investigate), > 0.25 (re-validate thresholds) |
| repeat failures per device / IP / ASN | hour | > 3 failures/hour from one device or > 20 from one ASN |
| share of a single signal in high-risk sessions | week | one signal group drives > 60% of `risk_high` (new attack tooling or broken sensor path) |
| review outcome: confirmed fraud rate | week | < 2% (thresholds too tight) or > 30% (τ_deny too lenient) |
| FaceTec `matchLevel` distribution | week | share of levels 4–5 rises > 50% vs baseline |

## Manual-review checklist

1. Reason code and top contributing signals (from `fdemo scan` output or the risk log).
2. Audit-trail image vs. ID photo: same person? (Audit Trail images may not be used to train/test ML — review only.)
3. Face boundary: colour/texture seams, mismatched skin tone between face and neck, flickering jaw/ears.
4. Screen or print cues: moiré, bezels, glare, paper edges, uniform focus.
5. Stream cues: duplicated/frozen frames, unnatural smoothness, frame rate below the device's camera.
6. Session context: new device, emulator/root flags, VPN/hosting ASN, velocity history.
7. Decide: approve / reject / step-up. Record the label — it feeds threshold re-validation.
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out
