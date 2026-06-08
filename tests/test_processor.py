import math

import pandas as pd
import pytest

from src.processor import SignalProcessor


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_df(close_values):
    """Return a minimal DataFrame with only a 'close' column."""
    return pd.DataFrame({"close": close_values}, dtype=float)


# ---------------------------------------------------------------------------
# Task 4.1 — process() returns a DataFrame
# ---------------------------------------------------------------------------

def test_process_returns_dataframe():
    """process() must return a pandas DataFrame."""
    df = make_df([1.0, 2.0, 3.0, 4.0, 5.0])
    result = SignalProcessor.process(df, window=3)
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Task 4.2 — rolling_mean column is added
# ---------------------------------------------------------------------------

def test_rolling_mean_column_present():
    """rolling_mean column is added to the DataFrame."""
    df = make_df([10.0, 20.0, 30.0, 40.0, 50.0])
    SignalProcessor.process(df, window=3)
    assert "rolling_mean" in df.columns


def test_rolling_mean_values_correct():
    """rolling_mean values match pandas rolling(window).mean()."""
    close = [10.0, 20.0, 30.0, 40.0, 50.0]
    df = make_df(close)
    SignalProcessor.process(df, window=3)
    expected = pd.Series(close, dtype=float).rolling(window=3).mean()
    pd.testing.assert_series_equal(df["rolling_mean"].reset_index(drop=True),
                                   expected.reset_index(drop=True),
                                   check_names=False)


# ---------------------------------------------------------------------------
# Task 4.3 — leading NaNs are preserved
# ---------------------------------------------------------------------------

def test_leading_nans_preserved_window3():
    """For window=3, first 2 rolling_mean entries must be NaN."""
    df = make_df([1.0, 2.0, 3.0, 4.0, 5.0])
    SignalProcessor.process(df, window=3)
    assert math.isnan(df["rolling_mean"].iloc[0])
    assert math.isnan(df["rolling_mean"].iloc[1])
    assert not math.isnan(df["rolling_mean"].iloc[2])


def test_leading_nans_preserved_window5():
    """For window=5, first 4 rolling_mean entries must be NaN (Req 4.3)."""
    df = make_df([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    SignalProcessor.process(df, window=5)
    for i in range(4):
        assert math.isnan(df["rolling_mean"].iloc[i]), (
            f"Expected NaN at index {i} but got {df['rolling_mean'].iloc[i]}"
        )
    for i in range(4, 7):
        assert not math.isnan(df["rolling_mean"].iloc[i]), (
            f"Expected non-NaN at index {i} but got {df['rolling_mean'].iloc[i]}"
        )


# ---------------------------------------------------------------------------
# Task 4.4 — signal column is added
# ---------------------------------------------------------------------------

def test_signal_column_present():
    """signal column is added to the DataFrame."""
    df = make_df([10.0, 20.0, 30.0, 40.0, 50.0])
    SignalProcessor.process(df, window=3)
    assert "signal" in df.columns


# ---------------------------------------------------------------------------
# Task 4.5 — signal == 0 for rows where rolling_mean is NaN (Req 5.4)
# ---------------------------------------------------------------------------

def test_signal_zero_when_rolling_mean_nan():
    """signal must be 0 for every row where rolling_mean is NaN."""
    df = make_df([100.0, 200.0, 300.0, 400.0, 500.0])
    SignalProcessor.process(df, window=3)
    nan_mask = df["rolling_mean"].isna()
    assert nan_mask.sum() > 0, "Expected some NaN rolling_mean rows"
    assert (df.loc[nan_mask, "signal"] == 0).all()


# ---------------------------------------------------------------------------
# Known-value end-to-end check (Req 4.1–4.3, 5.1–5.4, 11.6)
# ---------------------------------------------------------------------------

def test_signal_processor_known_values():
    """
    close = [1, 2, 3, 4, 5], window = 3
    rolling_mean = [NaN, NaN, 2.0, 3.0, 4.0]
    signal       = [  0,   0,   1,   1,   1]
      row 0: NaN  → signal 0
      row 1: NaN  → signal 0
      row 2: 3 > 2.0 → signal 1
      row 3: 4 > 3.0 → signal 1
      row 4: 5 > 4.0 → signal 1
    """
    df = make_df([1.0, 2.0, 3.0, 4.0, 5.0])
    result = SignalProcessor.process(df, window=3)

    assert math.isnan(result["rolling_mean"].iloc[0])
    assert math.isnan(result["rolling_mean"].iloc[1])
    assert result["rolling_mean"].iloc[2] == pytest.approx(2.0)
    assert result["rolling_mean"].iloc[3] == pytest.approx(3.0)
    assert result["rolling_mean"].iloc[4] == pytest.approx(4.0)

    assert list(result["signal"]) == [0, 0, 1, 1, 1]


def test_signal_zero_when_close_equals_rolling_mean():
    """signal must be 0 when close == rolling_mean (not strictly greater, Req 5.3)."""
    # With window=1, rolling_mean == close for every row.
    df = make_df([10.0, 20.0, 30.0])
    SignalProcessor.process(df, window=1)
    # close > rolling_mean is False for all rows (equal, not greater)
    assert (df["signal"] == 0).all()


def test_signal_binary_values():
    """All signal values must be 0 or 1 — no nulls or other values (Req 5.1)."""
    close = [10.0, 5.0, 15.0, 8.0, 20.0, 3.0, 12.0]
    df = make_df(close)
    SignalProcessor.process(df, window=3)
    assert df["signal"].isin([0, 1]).all()
    assert df["signal"].notna().all()


def test_window_respected():
    """SignalProcessor must use the window parameter, not a hardcoded value (Req 4.4)."""
    close = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    df2 = make_df(close)
    df4 = make_df(close)
    SignalProcessor.process(df2, window=2)
    SignalProcessor.process(df4, window=4)

    # window=2 → 1 NaN at start; window=4 → 3 NaNs at start
    assert df2["rolling_mean"].isna().sum() == 1
    assert df4["rolling_mean"].isna().sum() == 3
