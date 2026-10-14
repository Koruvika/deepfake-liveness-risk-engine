"""ISO/IEC 30107-3 style summaries of protocol results (decisions at the dev-chosen tau_op)."""
import numpy as np
import pandas as pd

from fdemo.eval.metrics import (
    acer, apcer_at_bpcer, auc, bpcer_at_apcer, clopper_pearson, cluster_bootstrap_ci, ibeta_indicative, rate_with_ci,
)


def accepted(res) -> np.ndarray:
    return (res["score"] < res["tau_op"]).to_numpy()


def flagged(res) -> np.ndarray:
    return (res["score"] >= res["tau_op"]).to_numpy()


def attack_table(res) -> pd.DataFrame:
    b = rate_with_ci(flagged(res[res.label == 0]))
    rows = []
    for attack, g in res[res.label == 1].groupby("attack"):
        errors = accepted(g)
        r = rate_with_ci(errors)
        iv = ibeta_indicative(r["k"], r["n"], b["rate"])
        boot_lo, boot_hi = cluster_bootstrap_ci(errors, g["actor"].to_numpy())
        rows.append({"attack": attack, "apcer": r["rate"], "k": r["k"], "n": r["n"], "lo": r["lo"], "hi": r["hi"],
                     "boot_lo": boot_lo, "boot_hi": boot_hi, "ibeta_L1": iv["L1"], "ibeta_L2": iv["L2"]})
    return pd.DataFrame(rows, columns=["attack", "apcer", "k", "n", "lo", "hi", "boot_lo", "boot_hi",
                                        "ibeta_L1", "ibeta_L2"])


def cluster_bootstrap_stat(df, stat_fn, cluster_col="actor", n_boot=2000, alpha=0.05, seed=0):
    """Joint percentile bootstrap CI for an arbitrary statistic of a dataframe, resampling whole clusters
    (e.g. actors) with replacement once per replicate.

    Use this (rather than averaging several independent `cluster_bootstrap_ci` intervals) when the
    statistic combines multiple correlated quantities -- e.g. a group rate that averages several
    attack-type rates which can share identities (actors) -- so every quantity is resampled consistently
    within a replicate instead of each being resampled independently.
    """
    clusters = df[cluster_col].to_numpy()
    uniq = np.unique(clusters)
    if len(uniq) == 0 or len(df) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx_by_cluster = {u: np.where(clusters == u)[0] for u in uniq}
    stats = np.empty(n_boot, float)
    for i in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_cluster[u] for u in sampled])
        stats[i] = stat_fn(df.iloc[idx])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _group_rate_stat(df) -> float:
    """Mean of per-attack-type accepted rates within df (the statistic jointly bootstrapped by actor)."""
    rates = [accepted(gg).mean() for _, gg in df.groupby("attack") if len(gg)]
    return float(np.mean(rates)) if rates else float("nan")


def _per_split_mean(res, fn, fallback_bona, fallback_att):
    """Average a curve metric per split (fold) instead of pooling scores across splits, which can mix
    incomparable score scales/distributions. Splits missing either class are skipped. Falls back to the
    pooled computation when there is only one split (or no "split" column)."""
    if "split" not in res.columns or res["split"].nunique() <= 1:
        return fn(fallback_bona, fallback_att)
    vals = []
    for _, g in res.groupby("split"):
        b, a = g[g.label == 0], g[g.label == 1]
        if len(b) == 0 or len(a) == 0:
            continue
        vals.append(fn(b.score, a.score))
    return float(np.mean(vals)) if vals else float("nan")


def overall(res) -> dict:
    bona, att = res[res.label == 0], res[res.label == 1]
    b = rate_with_ci(flagged(bona))
    tbl = attack_table(res)
    am = float(tbl.apcer.max()) if len(tbl) else float("nan")
    return {
        "bpcer": b["rate"], "bpcer_lo": b["lo"], "bpcer_hi": b["hi"], "apcer_max": am, "acer": acer(am, b["rate"]),
        "auc": _per_split_mean(res, auc, bona.score, att.score),
        "bpcer@apcer5": _per_split_mean(res, lambda b_, a_: bpcer_at_apcer(b_, a_, 0.05), bona.score, att.score),
        "apcer@bpcer1": _per_split_mean(res, lambda b_, a_: apcer_at_bpcer(b_, a_, 0.01), bona.score, att.score),
    }


def summarize_p2(res):
    res = res.assign(held=res["split"].str.split("/").str[0])
    groups, attacks = [], []
    for g, sub in res.groupby("held"):
        att_rows = sub[sub.label == 1]
        tbl = attack_table(sub)
        # group-level rate = mean of per-attack-type rates (not pooled), so a group with >1 attack type
        # (e.g. G2 = A2+A4) doesn't over-weight identities that appear in both attack types.
        group_apcer = float(tbl.apcer.mean()) if len(tbl) else float("nan")
        # Joint actor bootstrap of the group statistic itself: resample actors once per replicate and
        # recompute every attack type's rate on that same resample (not an average of independently
        # bootstrapped per-attack CIs), so attack types sharing identities are resampled consistently.
        # Enveloped with the Clopper-Pearson interval of the pooled group errors so this is never
        # narrower than the exact interval (see cluster_bootstrap_ci).
        joint_lo, joint_hi = cluster_bootstrap_stat(att_rows, _group_rate_stat, cluster_col="actor")
        pooled = rate_with_ci(accepted(att_rows))
        cp_lo, cp_hi = clopper_pearson(pooled["k"], pooled["n"])
        group_boot_lo = float(np.fmin(joint_lo, cp_lo))
        group_boot_hi = float(np.fmax(joint_hi, cp_hi))
        groups.append({"held": g, **overall(sub), "group_apcer": group_apcer,
                       "group_boot_lo": group_boot_lo, "group_boot_hi": group_boot_hi})
        attacks.append(tbl.assign(held=g))
    per_group = pd.DataFrame(groups)
    per_attack = pd.concat(attacks, ignore_index=True)
    am, bm = float(per_attack.apcer.max()), float(per_group.bpcer.mean())
    macro = {"apcer_max": am, "bpcer": bm, "acer": acer(am, bm), "auc": float(per_group.auc.mean()),
             "apcer@bpcer1": float(per_group["apcer@bpcer1"].mean())}
    return per_group, per_attack, macro


def summarize_p3(res) -> pd.DataFrame:
    """Per external video, majority vote across splits (accepted in > half the splits; ties count as
    accepted) before computing an exact Clopper-Pearson interval on the per-video verdicts."""
    rows = []
    bona = res[res.label == 0]
    bp = float(flagged(bona).mean()) if len(bona) else float("nan")
    ext = res[res.attack.str.startswith("EXT_")].copy()
    ext["accepted"] = accepted(ext)
    for attack, g in ext.groupby("attack"):
        per_vid = g.groupby("vid")["accepted"].apply(lambda s: bool(s.sum() * 2 >= len(s)))
        n = int(len(per_vid))
        k = int(per_vid.sum())
        lo, hi = clopper_pearson(k, n)
        rows.append({"attack": attack, "apcer": (k / n if n else float("nan")), "n": n, "lo": lo, "hi": hi,
                     "bpcer": bp})
    return pd.DataFrame(rows, columns=["attack", "apcer", "n", "lo", "hi", "bpcer"])


def success_check(groups_all, macro_all, groups_base, macro_base) -> dict:
    m = groups_all.merge(groups_base, on="held", suffixes=("_all", "_base"))
    nonoverlap = int((m.group_boot_hi_all < m.group_boot_lo_base).sum())
    acer_better = bool(macro_all["acer"] < macro_base["acer"])
    return {"acer_better": acer_better, "groups_nonoverlap": nonoverlap, "n_groups": int(len(m)),
            "passed": bool(acer_better and nonoverlap >= 3)}
