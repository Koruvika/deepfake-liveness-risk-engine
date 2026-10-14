"""Scorers for the 5 benchmark configurations.

B1/B2 are the open-source baselines used as-is (no training). B2+, F and ALL are learned fusions; the model
family (logistic regression vs. gradient boosting) is chosen by inner CV inside the training data, grouped by
the outer fold (`df["fold"]`, not `actor`) so a face-swap's source and target identities -- which can land in
different actors -- never split across the inner train/validation boundary."""
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PAD_COLS = ["pad_real_mean", "pad_real_min", "pad_real_low4", "pad_real_std"]
SBI_COLS = ["sbi_mean", "sbi_top4", "sbi_std"]
FORENSIC_PREFIXES = ("bn_", "sp_", "tm_", "id_", "rp_")
CONFIGS = ["B1", "B2", "B2+", "F", "ALL"]
RAW_CONFIGS = {"B1", "B2"}
CONFIG_LABELS = {
    "B1": "B1 MiniFASNet (passive PAD, open source)",
    "B2": "B2 SBI EfficientNet-B4 (deepfake, open source, mean over frames)",
    "B2+": "B2+ SBI + temporal aggregation (top-k, std)",
    "F": "F our forensic signals only",
    "ALL": "ALL fusion of everything (risk engine)",
}


def forensic_cols(df) -> list:
    return sorted(c for c in df.columns if c.startswith(FORENSIC_PREFIXES))


def config_columns(cfg, df) -> list:
    if cfg == "B1":
        return ["pad_real_mean"]
    if cfg == "B2":
        return ["sbi_mean"]
    if cfg == "B2+":
        return [c for c in SBI_COLS if c in df.columns]
    if cfg == "F":
        return forensic_cols(df)
    if cfg == "ALL":
        return [c for c in PAD_COLS + SBI_COLS if c in df.columns] + forensic_cols(df)
    raise ValueError(f"unknown config {cfg}")


def _raw(cfg, df) -> np.ndarray:
    if cfg == "B1":
        return 1.0 - df["pad_real_mean"].to_numpy(float)
    return df["sbi_mean"].to_numpy(float)


def candidates() -> dict:
    def imputer():
        return SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)

    return {
        "lr": Pipeline([("imp", imputer()), ("sc", StandardScaler()),
                        ("clf", LogisticRegression(C=0.5, class_weight="balanced", max_iter=5000))]),
        "hgb": Pipeline([("imp", imputer()),
                         ("clf", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                                                class_weight="balanced", random_state=0))]),
    }


def _cv(groups):
    return GroupKFold(n_splits=min(3, len(np.unique(groups))))


def _groups(df) -> np.ndarray:
    """Inner-CV grouping key: the outer fold, so identities spanning a face-swap pair stay together."""
    return df["fold"].to_numpy()


class Scorer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cols = []
        self.kind = "raw" if cfg in RAW_CONFIGS else None
        self.model = None
        self.fill = 0.5

    def fit(self, df):
        self.cols = config_columns(self.cfg, df)
        if self.cfg in RAW_CONFIGS:
            raw = _raw(self.cfg, df)
            self.fill = float(np.nanmedian(raw)) if np.isfinite(raw).any() else 0.5
            return self
        X, y, g = df[self.cols], df["label"].to_numpy(int), _groups(df)
        scores = {k: np.nanmean(cross_val_score(clone(est), X, y, groups=g, cv=_cv(g), scoring="roc_auc"))
                  for k, est in candidates().items()}
        self.kind = max(scores, key=scores.get)
        self.model = clone(candidates()[self.kind]).fit(X, y)
        return self

    def predict(self, df) -> np.ndarray:
        if self.cfg in RAW_CONFIGS:
            return np.nan_to_num(_raw(self.cfg, df), nan=self.fill)
        X = df.reindex(columns=self.cols)
        return self.model.predict_proba(X)[:, 1]

    def dev_scores(self, df) -> np.ndarray:
        """Out-of-fold scores on the training frame, used to pick thresholds without test leakage."""
        if self.cfg in RAW_CONFIGS:
            return self.predict(df)
        g = _groups(df)
        return cross_val_predict(clone(candidates()[self.kind]), df[self.cols], df["label"].to_numpy(int),
                                 groups=g, cv=_cv(g), method="predict_proba")[:, 1]
