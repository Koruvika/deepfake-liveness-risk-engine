"""Run every configuration × protocol, summarise, plot, write the benchmark report and the final model."""
import json
import logging

import joblib
import numpy as np
import pandas as pd

from fdemo import config
from fdemo.eval import report as R
from fdemo.eval.end_to_end import bpcer_breakdown, e2e_table
from fdemo.eval.metrics import apcer_at_bpcer
from fdemo.eval.protocols import OP_BPCER, p1_splits, p2_splits, p3_splits, run_protocol
from fdemo.eval.summary import attack_table, overall, success_check, summarize_p2, summarize_p3
from fdemo.fusion.failures import nan_free_cols
from fdemo.fusion.model import CONFIG_LABELS, CONFIGS, PAD_COLS, SBI_COLS, Scorer, config_columns
from fdemo.fusion.policy import make_thresholds
from fdemo.signals import SIGNAL_PREFIX

log = logging.getLogger(__name__)

ABLATION_GROUPS = SIGNAL_PREFIX


BASELINE_NAN_LIMIT = 0.20


def load_dataset():
    """Merge the manifest with extracted features. Any manifest row without a matching features row is
    dropped by the inner merge (reported in the Setup section).

    The two open-source baselines (MiniFASNet -> pad_real_mean, SBI -> sbi_mean) are compared against the
    risk engine throughout the report; if either baseline is dead -- its columns entirely missing from the
    features file, or NaN on more than 20% of non-EXT rows -- that comparison is meaningless, so this raises
    SystemExit instead of silently producing a bogus report. Any *other* (non-baseline) required column that
    is entirely missing is still reindexed back in as NaN, with a loud warning, so a partial extract run for
    signals outside the baselines doesn't crash the whole benchmark."""
    m = pd.read_csv(config.MANIFEST)
    f = pd.read_csv(config.FEATURES)
    df = m.merge(f, on="vid", how="inner")
    n_no_features = len(m) - len(df)
    baseline_cols = set(PAD_COLS) | set(SBI_COLS)
    required = list(dict.fromkeys(PAD_COLS + SBI_COLS))
    missing = [c for c in required if c not in df.columns]
    missing_baseline = [c for c in missing if c in baseline_cols]
    missing_other = [c for c in missing if c not in baseline_cols]
    if missing_baseline:
        raise SystemExit(
            f"FATAL: baseline columns missing from {config.FEATURES}: {missing_baseline}. The MiniFASNet/SBI "
            "baselines are dead, so the risk-engine-vs-baseline comparison would be meaningless. Re-run "
            "`fdemo extract` to regenerate features before benchmarking."
        )
    if missing_other:
        msg = f"WARNING: feature columns missing from {config.FEATURES}, filling with NaN: {missing_other}"
        log.warning(msg)
        print(msg)
        for c in missing_other:
            df[c] = np.nan

    nf = df["no_face"].astype(str).str.lower().isin(["true", "1"]) if len(df) else pd.Series([], dtype=bool)
    # no_face rows are excluded from every APCER/BPCER denominator downstream (they're routed to review, not
    # scored), so a baseline being NaN on them is expected (no face to score) and must not trip this check.
    non_ext = ((df["group"] != "EXT") & ~nf) if len(df) else pd.Series([], dtype=bool)
    for col in ("pad_real_mean", "sbi_mean"):
        if non_ext.any():
            nan_rate = float(df.loc[non_ext, col].isna().mean())
            if nan_rate > BASELINE_NAN_LIMIT:
                raise SystemExit(
                    f"FATAL: {col} is NaN on {nan_rate:.0%} of non-EXT, non-no_face rows (> "
                    f"{BASELINE_NAN_LIMIT:.0%}). This baseline is effectively dead, so the risk-engine-vs-"
                    "baseline comparison would be meaningless. Re-run `fdemo extract` to regenerate features "
                    "before benchmarking."
                )

    return df[~nf].reset_index(drop=True), df[nf], n_no_features


def run_all(df) -> dict:
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    out = {}
    for proto, fn in (("P1", p1_splits), ("P2", p2_splits), ("P3", p3_splits)):
        splits = fn(df)
        if not splits:
            continue
        for cfg in CONFIGS:
            res = run_protocol(df, cfg, splits)
            res.to_csv(config.RESULTS / f"{proto}_{cfg}.csv", index=False)
            out[(proto, cfg)] = res
    return out


def ablation(df, base_macro) -> pd.DataFrame:
    """base_macro: the already-computed P2/ALL macro summary (`p2["ALL"][2]`), reused for the "(none)" row
    instead of re-running the protocol a second time."""
    def score(frame):
        _, _, macro = summarize_p2(run_protocol(frame, "ALL", p2_splits(frame)))
        return macro["apcer@bpcer1"], macro["acer"]

    base_a, base_c = base_macro["apcer@bpcer1"], base_macro["acer"]
    rows = [{"removed": "(none)", "APCER@BPCER1%": base_a, "ACER": base_c, "Δ APCER@BPCER1%": 0.0}]
    for name, prefix in ABLATION_GROUPS.items():
        a, c = score(df.drop(columns=[c for c in df.columns if c.startswith(prefix)]))
        rows.append({"removed": name, "APCER@BPCER1%": a, "ACER": c, "Δ APCER@BPCER1%": a - base_a})
    return pd.DataFrame(rows)


def latency(df) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith("lat_")]
    if not cols:
        return pd.DataFrame(columns=["stage", "mean s/video"])
    return pd.DataFrame({"stage": [c[4:] for c in cols], "mean s/video": [df[c].mean() for c in cols]})


def train_final(df) -> None:
    syn = df[df.group != "EXT"]
    scorer = Scorer("ALL").fit(syn)
    dev = scorer.dev_scores(syn)
    y = syn.label.to_numpy()
    th = make_thresholds(dev[y == 0], dev[y == 1])
    bundle = {"scorer": scorer, "thresholds": th,
              "medians": syn[scorer.cols].median().to_dict(), "columns": scorer.cols,
              "nan_free_cols": nan_free_cols(syn, scorer.cols)}
    config.BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, config.BUNDLE)
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "final_thresholds.json").write_text(json.dumps({"review": th.review, "deny": th.deny}))


IBETA_FOOTNOTE = "\\* fewer than 150 attempts: an iBeta-style verdict is not attainable."


def _ibeta_cell(r) -> str:
    cell = f"{'pass' if r.ibeta_L1 else 'fail'} / {'pass' if r.ibeta_L2 else 'fail'}"
    if r.n < config.N_ATTACK:
        cell += f"* (n={r.n} < {config.N_ATTACK})"
    return cell


def _attack_rows(tbl):
    t = tbl.copy()
    t["APCER (95% CI: exact / actor-bootstrap)"] = [
        f"{R.pct(r.apcer)} ({r.k}/{r.n}; {R.pct(r.lo)}–{R.pct(r.hi)} / {R.pct(r.boot_lo)}–{R.pct(r.boot_hi)})"
        for r in t.itertuples()]
    t["standard"] = [config.ATTACK_INFO.get(a, ("", "external generator"))[1] for a in t.attack]
    t["iBeta-style L1 / L2 (indicative)"] = [_ibeta_cell(r) for r in t.itertuples()]
    return t[["attack", "standard", "APCER (95% CI: exact / actor-bootstrap)", "iBeta-style L1 / L2 (indicative)"]]


def _group_table(p2, cfgs) -> pd.DataFrame:
    """Per-held-group APCER (mean-of-attack-types rate with actor-bootstrap CI), BPCER and AUC."""
    rows = []
    for cfg in cfgs:
        per_group = p2[cfg][0]
        for r in per_group.itertuples():
            rows.append({"config": CONFIG_LABELS[cfg], "held": r.held, "group_apcer": r.group_apcer,
                         "group_boot_lo": r.group_boot_lo, "group_boot_hi": r.group_boot_hi,
                         "bpcer": r.bpcer, "auc": r.auc})
    return pd.DataFrame(rows)


def signal_coverage(df, dropped) -> pd.DataFrame:
    """Per attack (including bona and EXT_*), the share of videos where every feature of a signal group is
    NaN, plus how many videos of that attack were dropped for no_face."""
    all_df = pd.concat([df, dropped], ignore_index=True)
    rows = []
    for attack, g in all_df.groupby("attack"):
        row = {"attack": attack}
        for name, prefix in SIGNAL_PREFIX.items():
            cols = [c for c in g.columns if c.startswith(prefix)]
            row[name] = float(g[cols].isna().all(axis=1).mean()) if cols else float("nan")
        rows.append(row)
    cov = pd.DataFrame(rows)
    dropped_n = dropped.groupby("attack").size() if len(dropped) else pd.Series(dtype=int)
    cov["no_face dropped"] = cov["attack"].map(dropped_n).fillna(0).astype(int)
    return cov


def run_bench():
    df, dropped, n_no_feat = load_dataset()
    results = run_all(df)
    fig = config.FIGURES

    p1 = {cfg: overall(results[("P1", cfg)]) for cfg in CONFIGS}
    p2 = {cfg: summarize_p2(results[("P2", cfg)]) for cfg in CONFIGS}
    base = min(("B1", "B2"), key=lambda c: p2[c][2]["acer"])
    success = success_check(p2["ALL"][0], p2["ALL"][2], p2[base][0], p2[base][2])
    p3 = {cfg: summarize_p3(results[("P3", cfg)]) for cfg in CONFIGS if ("P3", cfg) in results}
    if p3:
        # P3's capture differs from P1/P2 (square fakes composited onto a blurred 16:9 background): a
        # temporal/background-reactive signal could pick up the padding itself rather than the attack.
        # "ALL-tm" = ALL with tm_* (temporal) features dropped, run on the same P3 splits, shown next to ALL
        # so the reader can see how much of ALL's P3 performance the temporal group is responsible for.
        tm_cols = [c for c in df.columns if c.startswith("tm_")]
        df_no_tm = df.drop(columns=tm_cols)
        p3["ALL−tm"] = summarize_p3(run_protocol(df_no_tm, "ALL", p3_splits(df_no_tm)))

    # Same rule `train_final` uses for `bundle["nan_free_cols"]`: computed over the synthetic (non-EXT) rows,
    # over the ALL config's feature columns, so scan and the e2e tables can't disagree about what a signal
    # failure is.
    nfc = nan_free_cols(df[df.group != "EXT"], config_columns("ALL", df))
    err_cols = [c for c in df.columns if c.startswith("err_")]
    feats = df[["vid"] + list(dict.fromkeys(["pad_real_mean"] + err_cols + nfc))]
    e2e = e2e_table(results[("P2", "ALL")], feats, nan_free_cols=nfc)
    bbd = bpcer_breakdown(results[("P2", "ALL")], feats, nan_free_cols=nfc)
    abl = ablation(df, p2["ALL"][2])

    apcer_by_cfg = {c: p2[c][1].groupby("attack", as_index=False).apcer.max() for c in ("B1", "B2", "ALL")}
    R.plot_tradeoff({c: results[("P1", c)] for c in CONFIGS}, fig / "roc_p1.png")
    R.plot_apcer_bars(apcer_by_cfg, fig / "apcer_p2.png")
    R.plot_risk_hist(results[("P1", "ALL")], fig / "risk_hist_p1.png")
    R.plot_e2e(e2e, fig / "e2e_bypassed.png")

    # Same frames as the apcer_p2.png bar chart, as a table (figure/table parity).
    apcer_wide = apcer_by_cfg["B1"].rename(columns={"apcer": "B1"})
    for c in ("B2", "ALL"):
        apcer_wide = apcer_wide.merge(apcer_by_cfg[c].rename(columns={"apcer": c}), on="attack", how="outer")
    p1_res = results[("P1", "ALL")]
    p1_tau_text = f"P1 median τ_review = {p1_res.tau_review.median():.3f}, τ_deny = {p1_res.tau_deny.median():.3f}."

    train_final(df)

    counts = df.groupby("attack").size().rename("videos").reset_index()
    attacks_std = pd.DataFrame([{"attack": a, "description": d, "standard": s} for a, (d, s) in config.ATTACK_INFO.items()])
    p1_tbl = pd.DataFrame([{"config": CONFIG_LABELS[c], **{k: p1[c][k] for k in ("auc", "apcer_max", "bpcer", "acer",
                           "bpcer@apcer5", "apcer@bpcer1")}} for c in CONFIGS])
    p2_tbl = pd.DataFrame([{"config": CONFIG_LABELS[c], **p2[c][2]} for c in CONFIGS])
    pct_cols = {k: R.pct for k in ("apcer_max", "bpcer", "acer", "bpcer@apcer5", "apcer@bpcer1")}
    b1_p2 = p2["B1"][1].groupby("attack", as_index=False).apcer.max()
    b1_lines = "\n".join(f"- {a}: accepted {R.pct(v)} of attacks" for a, v in zip(b1_p2.attack, b1_p2.apcer))
    e2e_tbl = e2e.drop(columns=["n"]).assign(allow=e2e.allow.map(R.pct), review=e2e.review.map(R.pct),
                                             deny=e2e.deny.map(R.pct))
    p3_text = "Not run: TalkingHeadBench could not be fetched (P3 skipped)."
    if p3:
        p3_text = "\n\n".join(f"**{c}**\n\n" + R.md_table(t, {"apcer": R.pct, "lo": R.pct, "hi": R.pct, "bpcer": R.pct})
                              for c, t in p3.items() if c in ("B1", "B2", "ALL", "ALL−tm"))
    p2_attack_src = p2["ALL"][1]
    p2_attack_tbl = _attack_rows(p2_attack_src)
    p2_ibeta_footnote = IBETA_FOOTNOTE if (p2_attack_src.n < config.N_ATTACK).any() else ""
    p1_ibeta_src = attack_table(results[("P1", "ALL")])
    p1_attack_tbl = _attack_rows(p1_ibeta_src)
    p1_ibeta_footnote = IBETA_FOOTNOTE if (p1_ibeta_src.n < config.N_ATTACK).any() else ""
    group_tbl = _group_table(p2, ("ALL", base))
    # auc is a discriminability score, not a rate -- render it as a plain 3-decimal number (md_table's
    # default float format) like p1_tbl/p2_tbl do, not as a percentage.
    group_pct_cols = {k: R.pct for k in ("group_apcer", "group_boot_lo", "group_boot_hi", "bpcer")}
    cov = signal_coverage(df, dropped)
    head = (f"ALL vs best open-source baseline ({base}) on P2: ACER {R.pct(p2['ALL'][2]['acer'])} vs "
            f"{R.pct(p2[base][2]['acer'])}; APCER@BPCER1% {R.pct(p2['ALL'][2]['apcer@bpcer1'])} vs "
            f"{R.pct(p2[base][2]['apcer@bpcer1'])}. Success criterion (lower ACER and non-overlapping 95% "
            f"actor-bootstrap CIs in ≥3 of {success['n_groups']} held-out groups): "
            f"**{'met' if success['passed'] else 'NOT met'}** "
            f"({success['groups_nonoverlap']}/{success['n_groups']} groups non-overlapping).")

    text = f"""# Benchmark report: multi-layer deepfake / spoof / injection risk engine (CPU demo)

## 1. Summary

{head}

## 2. Setup

- Bona fide: RAVDESS, 12 actors (6 male, 6 female), 15 clips each, 3.2 s @ 25 fps. Identity-disjoint 4-fold split;
  face-swap sources always come from the same fold as the target.
- Bona fide and A1–A5 all pass through the same "SDK capture" encode (640×360, 25 fps, 80 frames, H.264 CRF 23),
  including the same decode/re-encode path before it (read frames -> CRF 12 re-encode -> SDK capture), so no
  detector can key on codec, size, frame rate or an extra generation step. EXT (P3, TalkingHeadBench) uses a
  padded variant instead (see §5 / §12): square fakes composited onto a blurred 16:9 background.
- Videos dropped because no face was found in >50% of frames: {len(dropped)}. Manifest rows with no matching
  extracted features (dropped by the manifest/features inner merge): {n_no_feat}. no_face videos are excluded
  from every APCER/BPCER denominator in this report; in deployment they are routed to review, not auto-allowed.
- Decisions in the per-attack tables use τ_op chosen on development data (out-of-fold, training folds only) at
  BPCER = {R.pct(OP_BPCER)}; curve metrics (AUC, BPCER@APCER5%, APCER@BPCER1%) are threshold-free.
- Baselines keep their reference preprocessing (MiniFASNet scale 2.7/4.0 crops; SBI test-time crop). Only deviation:
  RetinaFace-MobileNet (uniface) replaces RetinaFace-ResNet50 for SBI, for CPU speed.

**Statistics**

- Exact Clopper–Pearson CIs assume independent attempts; with 12 identities and A4 derived from A2 they are
  anti-conservative, so the "actor-bootstrap" interval shown alongside them is the envelope of the actor-cluster
  bootstrap (2,000 resamples) and the Clopper–Pearson interval (never narrower than either), and the success
  criterion (§1) uses it. Clusters are target identities; face-swap source identities (A1/A2) share folds with
  their targets but are not separately clustered in the bootstrap — a residual source of anti-conservatism we
  disclose rather than correct for.
- AUC, BPCER@APCER5% and APCER@BPCER1% are computed per split and averaged; the two DET points are a-posteriori
  (thresholds set on test scores) — not deployable operating points. Deployable decisions use τ chosen on
  development data (§4 recommended operating point, `fdemo.eval.recommend`).
- ACER follows ISO/IEC 30107-3:2017; the 2023 revision deprecates it, so APCER_max and BPCER are the headline
  numbers.
- P3: each external video is scored by the 4 fold models and counted by majority vote (ties count as accepted,
  i.e. the conservative direction for APCER). Per-fold models are used rather than one pooled model because bona
  fide must stay identity-disjoint between training and evaluation; P3's bona fide come from the same held-out
  fold as P1/P2.

{R.md_table(attacks_std)}

{R.md_table(counts)}

## 3. P1 — seen attacks (identity-disjoint 4-fold)

{R.md_table(p1_tbl, pct_cols)}

![P1 trade-off](figures/roc_p1.png)

![Risk distribution](figures/risk_hist_p1.png)

{p1_tau_text}

## 4. P2 — unseen attack group (leave-one-group-out × identity folds; A2+A4 held out together)

Macro over the four held-out groups:

{R.md_table(p2_tbl, pct_cols)}

Per held-out group, risk engine (ALL) vs best open-source baseline ({base}):

{R.md_table(group_tbl, group_pct_cols)}

Per attack type, risk engine (ALL):

{R.md_table(p2_attack_tbl)}

{p2_ibeta_footnote}

![P2 APCER per attack](figures/apcer_p2.png)

Same data as the figure above, as a table (APCER at dev-chosen BPCER 5%):

{R.md_table(apcer_wide, {c: R.pct for c in ("B1", "B2", "ALL")})}

## 5. P3 — external generators (TalkingHeadBench fakes only; APCER at τ_op)

TalkingHeadBench's capture differs from P1/P2's: it ships square fakes composited onto a blurred 16:9
background, not an end-to-end SDK-captured clip of the same framing as our bona fide/attack videos. A
temporal or background-reactive signal (e.g. our `tm_*` group) could key on the padding itself rather than
on the attack — every EXT fake carries that padding, so such a signal would flag it reliably regardless of
whether the face is a deepfake, *deflating* P3 APCER (making detection look better than it is) for reasons
unrelated to deepfake detection. "ALL−tm" below is the risk engine with `tm_*` features dropped, on the same
P3 splits, so the reader can see how much of ALL's (understated) P3 APCER the temporal group is responsible
for.

{p3_text}

## 6. iBeta-style indicative verdicts (P1, risk engine)

Indicative only — iBeta tests physical artefacts presented by people to a real sensor; these are digital
simulations. L1: 0 successful attacks per species (150 attempts), BPCER ≤ 15%. L2: APCER ≤ 1%, BPCER ≤ 15%.

{R.md_table(p1_attack_tbl)}

{p1_ibeta_footnote}

## 7. How a typical passive liveness SDK responds

B1 (MiniFASNet, Apache-2.0, the most widely used open-source passive liveness model) on attack groups it was
never tuned for (P2, at its dev-chosen τ_op):

{b1_lines}

ISO/IEC 30107-3 PAD testing covers presentation attacks at the sensor; digital face swaps and virtual-camera
injection (CEN/TS 18099) are out of its scope, which is exactly where a passive PAD model is weakest.

## 8. End-to-end system response

Decision rates per attack under three vendor scenarios (vendor stand-in = MiniFASNet, **not FaceTec**):

{R.md_table(e2e_tbl)}

Genuine-user outcomes under vendor + engine (who causes the rejections):

{R.md_table(bbd, {k: R.pct for k in ("deny_by_vendor", "deny_by_engine", "review_signal_failure", "review", "allow")})}

![End-to-end, vendor bypassed](figures/e2e_bypassed.png)

## 9. Ablation (P2, risk engine without one signal group)

{R.md_table(abl, {"APCER@BPCER1%": R.pct, "ACER": R.pct, "Δ APCER@BPCER1%": lambda v: f"{100 * v:+.1f} pp"})}

Δ > 0: removing the group lets more attacks through (the group helps).

## 10. Latency (CPU, mean seconds per 80-frame video)

{R.md_table(latency(df))}

## 11. Signal coverage

Share of videos, per attack type, where every feature of a signal group is NaN (signal failed or was skipped),
and how many videos of that attack type were dropped upstream for no_face. no_face videos are excluded from
every APCER/BPCER denominator elsewhere in this report (their routing to review in deployment is a policy
decision, not a detection result):

{R.md_table(cov, {name: R.pct for name in SIGNAL_PREFIX})}

A skewed missing rate can act as a shortcut through the imputer's missing-indicator (the fusion models impute
with `add_indicator=True`): if a signal group is systematically missing on one attack type but not on bona fide,
the model can learn to key on "this signal is absent" rather than on the forensic content of the other signals.

## 12. Limitations

- Attacks are generated by us; real attackers iterate against the deployed system, so real-world APCER will be higher.
- Bona fide videos are compressed studio recordings (RAVDESS), not live webcam streams; sensor-noise cues are weaker
  than on a raw capture path.
- No real virtual-camera / emulator / hardware-injection test and no device-integrity signal (specified, not built:
  see `vulnerability_assessment_sample.md` §c).
- P3 (TalkingHeadBench) capture differs from P1/P2: square fakes composited onto a blurred 16:9 background, not
  the same SDK-captured framing as our bona fide/attacks. Temporal and background-reactive signals may key on
  that padding rather than the attack itself, which would *deflate* P3 APCER (make detection look better than
  it is, not worse) — see §5's "ALL−tm" row.
- The vendor is a stand-in (MiniFASNet), not FaceTec; the end-to-end table shows the integration logic only.
- iBeta-style verdicts are indicative; sample sizes give the 95% intervals shown.
- SBI weights, inswapper and RAVDESS are research / non-commercial assets — this demo is not a production model.
- Exact (Clopper–Pearson) confidence intervals assume independent attempts; ours are clustered by actor (and
  A4 is derived from A2), so they understate uncertainty — read the actor-bootstrap interval alongside them.
- The FaceTec vendor adapter only parses liveness from v10 (`result.livenessProven`); v9 (`/match-3d-2d-idscan`)
  responses are treated as match-level only, since we could not verify a liveness field on v9 payloads.

## 13. Next steps

Session/device integrity signals (Play Integrity, App Attest, virtual-camera and timestamp-jitter checks), an active
colour-flash challenge, real virtual-camera and emulator tests on the client's staging stack, and re-fitting the
fusion on the client's labelled FaceTec sessions.
"""
    out = config.REPORTS / "benchmark_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out
