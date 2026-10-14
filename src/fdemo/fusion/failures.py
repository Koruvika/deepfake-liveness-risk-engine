"""Shared "a signal's output can't be trusted" rule, used identically by `fdemo scan` and the eval
end-to-end tables so the two can't drift apart.

A NaN in a feature column is not automatically suspicious: the fusion imputer's median-imputation plus
missing-indicator (`add_indicator=True`) is exactly how the model was trained to handle features that are
sometimes missing (e.g. a signal that legitimately fails on some fraction of videos). What *is* suspicious is
a NaN in a column that had **zero** NaN in the training data -- the model never learned what to do with a
missing value there, so the imputed value is pure guesswork, not a trained fallback."""
import numpy as np
import pandas as pd


def nan_free_cols(df, cols) -> list:
    """The subset of `cols` that have zero NaN in `df`. Compute this once on the training frame (bench's
    `train_final`) and reuse the exact same list at scan/e2e time -- the rule must not drift."""
    return [c for c in cols if c in df.columns and df[c].notna().all()]


def signal_failure_mask(features_df, nan_free_cols) -> np.ndarray:
    """True for a row if any `err_*` flag is truthy, or any column in `nan_free_cols` is NaN for that row."""
    n = len(features_df)
    failure = np.zeros(n, dtype=bool)
    for c in features_df.columns:
        if c.startswith("err_"):
            failure |= features_df[c].fillna(0).astype(bool).to_numpy()
    for c in nan_free_cols:
        if c in features_df.columns:
            failure |= pd.isna(features_df[c]).to_numpy()
    return failure
