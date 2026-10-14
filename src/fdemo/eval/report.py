"""Markdown helpers and figures for the benchmark report."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"B1": "#2a78d6", "B2": "#eb6834", "B2+": "#1baf7a", "F": "#eda100", "ALL": "#e87ba4"}
STATUS = {"allow": "#0ca30c", "review": "#fab219", "deny": "#d03b3b"}


def pct(v) -> str:
    return "—" if pd.isna(v) else f"{100 * v:.1f}%"


def md_table(df, fmt=None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt:
                cells.append(fmt[c](v))
            elif isinstance(v, (float, np.floating)):
                cells.append("—" if np.isnan(v) else f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _axes(w=6.4, h=4.2, ncols=1):
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), dpi=150, facecolor=SURFACE, squeeze=False)
    for ax in axes.ravel():
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    return fig, axes.ravel()


def _finish(fig, path, title, top=None):
    """`top` reserves vertical figure space (as a tight_layout rect) for a figure-level legend placed above
    the panels, so it never overlaps the suptitle or the axes."""
    fig.suptitle(title, color=INK, fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, top) if top is not None else None)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_tradeoff(results_by_cfg, path):
    """APCER vs BPCER trade-off (lower-left is better), one line per configuration."""
    fig, (ax,) = _axes()
    for cfg, res in results_by_cfg.items():
        fpr, tpr, _ = roc_curve(res.label, res.score)
        keep = fpr > 0
        ax.plot(fpr[keep], 1 - tpr[keep], lw=2, color=SERIES[cfg], label=cfg)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("BPCER — genuine users rejected (log scale)", color=INK2, fontsize=8)
    ax.set_ylabel("APCER — attacks accepted", color=INK2, fontsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    return _finish(fig, path, "P1 (seen attacks): APCER vs BPCER trade-off (pooled over folds)")


def plot_apcer_bars(tables_by_cfg, path):
    attacks = sorted(set().union(*[set(t.attack) for t in tables_by_cfg.values()]))
    x = np.arange(len(attacks))
    width = 0.8 / len(tables_by_cfg)
    fig, (ax,) = _axes(7.2, 4.2)
    for i, (cfg, t) in enumerate(tables_by_cfg.items()):
        vals = [t.set_index("attack").apcer.get(a, np.nan) for a in attacks]
        ax.bar(x + (i - (len(tables_by_cfg) - 1) / 2) * width, vals, width, color=SERIES[cfg],
               edgecolor=SURFACE, linewidth=1.5, label=cfg)
    ax.set_xticks(x, attacks)
    ax.set_ylim(0, 1)
    ax.set_ylabel("APCER at dev-chosen BPCER 5% (lower is better)", color=INK2, fontsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, ncols=len(tables_by_cfg))
    return _finish(fig, path, "P2 (unseen attack group): APCER per attack type")


def plot_risk_hist(res, path):
    fig, (ax,) = _axes()
    bins = np.linspace(0, 1, 41)
    ax.hist(res[res.label == 0].score, bins=bins, histtype="step", lw=2, color=INK2, label="bona fide")
    ax.hist(res[res.label == 1].score, bins=bins, histtype="step", lw=2, color=STATUS["deny"], label="attacks")
    # Log scale: a tall top bin (e.g. a pile-up of attacks near risk=1) would otherwise flatten the bona
    # fide distribution into the x-axis. ylim bottom 0.8 (not 0) keeps empty bins (count 0) visibly at the
    # floor instead of being clipped off a log axis.
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.8)
    for tau, name in ((res.tau_review.median(), "review"), (res.tau_deny.median(), "deny")):
        ax.axvline(tau, color=INK2, lw=1, ls="--")
        # A fixed fraction of the axes height (not data-dependent ylim*0.95, which on a log axis would place
        # the label at an awkward height): x in data coords, y in axes-fraction coords.
        ax.text(tau, 0.95, f" {name} τ={tau:.2f}", color=INK2, fontsize=7, va="top",
                transform=ax.get_xaxis_transform())
    ax.set_xlabel("risk score (ALL)", color=INK2, fontsize=8)
    ax.set_ylabel("videos (log scale)", color=INK2, fontsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    return _finish(fig, path, "P1: risk score distribution with policy thresholds")


def plot_e2e(e2e, path, scenario="V-bypassed"):
    sub = e2e[e2e.scenario == scenario]
    attacks = sorted(sub.attack.unique(), key=lambda a: (a != "bona", a))
    fig, axes = _axes(8.0, 3.6, ncols=2)
    for ax, mode in zip(axes, ["vendor-only", "vendor+engine"]):
        m = sub[sub["mode"] == mode].set_index("attack").reindex(attacks)
        left = np.zeros(len(attacks))
        for act in ("allow", "review", "deny"):
            ax.barh(attacks, m[act], left=left, color=STATUS[act], edgecolor=SURFACE, linewidth=1.5, label=act)
            left += m[act].to_numpy()
        ax.set_xlim(0, 1)
        ax.set_title(mode, color=INK2, fontsize=9)
        ax.invert_yaxis()
    # A single figure-level legend above both panels: a per-axes legend (e.g. bottom-right of the second
    # panel) can overlap a bar (the A5 row in the right panel), which a legend placed outside the axes never
    # can.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=3, frameon=False, labelcolor=INK2, fontsize=8,
               bbox_to_anchor=(0.5, 0.93))
    return _finish(fig, path, f"End-to-end decisions, scenario {scenario} (vendor bypassed by injection)", top=0.82)
