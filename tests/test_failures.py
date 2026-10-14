import numpy as np
import pandas as pd

from fdemo.fusion.failures import nan_free_cols, signal_failure_mask


def test_nan_free_cols_excludes_columns_with_any_nan():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, float("nan"), 3.0], "c": [1.0, 2.0, 3.0]})
    assert nan_free_cols(df, ["a", "b", "c", "missing"]) == ["a", "c"]


def test_signal_failure_mask_truth_table():
    df = pd.DataFrame({
        "a": [1.0, float("nan"), 1.0, 1.0, float("nan")],
        "b": [1.0, 1.0, float("nan"), 1.0, 1.0],   # b had NaN at training time -> not in nan_free_cols
        "err_sig": [0, 0, 0, 1, 0],
    })
    mask = signal_failure_mask(df, nan_free_cols=["a"])
    # row0: no NaN in a, no err -> False
    # row1: NaN in a (a nan-free col) -> True
    # row2: NaN in b, but b is not a nan-free col -> False
    # row3: err_sig truthy -> True
    # row4: NaN in a -> True
    assert list(mask) == [False, True, False, True, True]


def test_signal_failure_mask_ignores_nan_in_columns_that_had_nan_at_training_time():
    """A NaN only matters if the column was nan-free in training; otherwise the model's imputer already
    learned to handle it, so it must not fire a signal_failure."""
    df = pd.DataFrame({"a": [float("nan")], "b": [1.0]})
    assert list(signal_failure_mask(df, nan_free_cols=["b"])) == [False]


def test_signal_failure_mask_no_err_no_nan_free_cols():
    df = pd.DataFrame({"a": [1.0, 2.0]})
    assert list(signal_failure_mask(df, nan_free_cols=[])) == [False, False]
