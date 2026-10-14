"""ISO/IEC 30107-3 style metrics. Higher score = more attack-like; attack if score >= tau."""
import numpy as np
from scipy.stats import beta
from sklearn.metrics import roc_auc_score


def apcer(attack_scores, tau) -> float:
    s = np.asarray(attack_scores, float)
    return float(np.mean(s < tau)) if len(s) else float("nan")


def bpcer(bona_scores, tau) -> float:
    s = np.asarray(bona_scores, float)
    return float(np.mean(s >= tau)) if len(s) else float("nan")


def threshold_at_bpcer(bona_scores, target) -> float:
    """Smallest tau with BPCER(tau) <= target."""
    s = np.sort(np.asarray(bona_scores, float))
    n = len(s)
    k = int(np.floor(target * n + 1e-9))
    if k >= n:
        return float("-inf")
    return float(np.nextafter(s[n - k - 1], np.inf))


def threshold_at_apcer(attack_scores, target) -> float:
    """Largest tau with APCER(tau) <= target."""
    s = np.sort(np.asarray(attack_scores, float))
    n = len(s)
    k = int(np.floor(target * n + 1e-9))
    if k >= n:
        return float("inf")
    return float(s[k])


def bpcer_at_apcer(bona_scores, attack_scores, target=0.05) -> float:
    return bpcer(bona_scores, threshold_at_apcer(attack_scores, target))


def apcer_at_bpcer(bona_scores, attack_scores, target=0.01) -> float:
    return apcer(attack_scores, threshold_at_bpcer(bona_scores, target))


def auc(bona_scores, attack_scores) -> float:
    if len(bona_scores) == 0 or len(attack_scores) == 0:
        return float("nan")
    y = np.r_[np.zeros(len(bona_scores)), np.ones(len(attack_scores))]
    s = np.r_[np.asarray(bona_scores, float), np.asarray(attack_scores, float)]
    return float(roc_auc_score(y, s))


def acer(apcer_max, bpcer_value) -> float:
    return (apcer_max + bpcer_value) / 2.0


def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def rate_with_ci(errors) -> dict:
    e = np.asarray(errors, bool)
    k, n = int(e.sum()), int(e.size)
    lo, hi = clopper_pearson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "lo": lo, "hi": hi}


def ibeta_indicative(k, n, bpcer_value) -> dict:
    """Indicative only: L1 = 0 successful attacks, L2 <= 1%; both cap BPCER at 15%."""
    ok_b = bpcer_value <= 0.15
    return {"L1": bool(n > 0 and k == 0 and ok_b), "L2": bool(n > 0 and k / n <= 0.01 and ok_b)}


def cluster_bootstrap_ci(errors, clusters, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for a rate, resampling whole clusters (e.g. actors) with replacement,
    enveloped with the exact (Clopper-Pearson) interval on the same errors.

    Never narrower than the exact interval: the percentile bootstrap can collapse to a point (e.g.
    [0, 0] when every cluster has zero errors, or a near-point interval with very few clusters) even
    though per-row independence doesn't hold and the true uncertainty is at least as large as the exact
    (i.i.d.) interval. Taking the envelope (min of the los, max of the his) guards against that collapse
    while still widening the interval when clustering adds genuine extra uncertainty.
    """
    e = np.asarray(errors, bool)
    c = np.asarray(clusters)
    uniq = np.unique(c)
    if len(uniq) == 0 or len(e) == 0:
        return float("nan"), float("nan")
    cp_lo, cp_hi = clopper_pearson(int(e.sum()), int(e.size))
    rng = np.random.default_rng(seed)
    idx_by_cluster = {u: np.where(c == u)[0] for u in uniq}
    rates = np.empty(n_boot, float)
    for i in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_cluster[u] for u in sampled])
        rates[i] = e[idx].mean()
    boot_lo, boot_hi = np.percentile(rates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.fmin(boot_lo, cp_lo)), float(np.fmax(boot_hi, cp_hi))
