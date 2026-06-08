"""Property-based tests using Hypothesis (Task 9).

Each test validates a correctness property defined in the design document.
"""

import json
import math
import sys
import unittest.mock

import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from src.config import Config, ConfigLoader
from src.metrics import MetricsWriter
from src.processor import SignalProcessor

import run


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

_seeds = st.integers(min_value=0, max_value=2**31 - 1)
_windows = st.integers(min_value=1, max_value=20)
_versions = st.text(
    min_size=1,
    max_size=10,
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"),
        whitelist_characters="_-",
        max_codepoint=127,  # restrict to ASCII to avoid file encoding issues
    ),
)
_close_floats = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)
_close_lists = st.lists(_close_floats, min_size=1, max_size=100)


# ---------------------------------------------------------------------------
# Property 1: Config round-trip field access (Req 2.1)
# ---------------------------------------------------------------------------

@given(seed=_seeds, window=_windows, version=_versions)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_config_roundtrip(tmp_path, seed, window, version):
    """**Validates: Requirements 2.1**

    For any valid (seed, window, version), ConfigLoader.load() returns a
    Config object where all three fields are accessible and equal to the
    values in the YAML.
    """
    cfg_file = tmp_path / f"config_{seed}_{window}.yaml"
    cfg_file.write_text(f"seed: {seed}\nwindow: {window}\nversion: '{version}'\n")

    config = ConfigLoader.load(str(cfg_file))

    assert config.seed == seed
    assert config.window == window
    assert config.version == version


# ---------------------------------------------------------------------------
# Property 2: ConfigLoader rejects missing fields (Req 2.2, 2.3, 2.4)
# ---------------------------------------------------------------------------

@given(
    seed=st.one_of(st.none(), _seeds),
    window=st.one_of(st.none(), _windows),
    version=st.one_of(st.none(), _versions),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_config_missing_field_raises(tmp_path, seed, window, version):
    """**Validates: Requirements 2.2, 2.3, 2.4**

    For any config missing one or more required fields, ConfigLoader.load()
    raises a ValueError.
    """
    # At least one field must be missing for this property to apply
    assume(seed is None or window is None or version is None)

    lines = []
    if seed is not None:
        lines.append(f"seed: {seed}")
    if window is not None:
        lines.append(f"window: {window}")
    if version is not None:
        lines.append(f"version: '{version}'")

    yaml_content = "\n".join(lines) + "\n" if lines else "{}\n"
    cfg_file = tmp_path / "config_missing.yaml"
    cfg_file.write_text(yaml_content)

    with pytest.raises(ValueError):
        ConfigLoader.load(str(cfg_file))


# ---------------------------------------------------------------------------
# Property 3: Rolling mean NaN prefix invariant (Req 4.2, 4.3)
# ---------------------------------------------------------------------------

@given(close_values=_close_lists, window=_windows)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_rolling_mean_nan_prefix(close_values, window):
    """**Validates: Requirements 4.2, 4.3**

    For any DataFrame with at least window rows, the first window-1 entries
    of rolling_mean are NaN, and all entries from index window-1 onward
    are non-NaN.
    """
    # Property only applies when there are at least window rows
    assume(len(close_values) >= window)

    df = pd.DataFrame({"close": close_values}, dtype=float)
    SignalProcessor.process(df, window)

    # First window-1 entries must be NaN
    for i in range(window - 1):
        assert math.isnan(df["rolling_mean"].iloc[i]), (
            f"Expected NaN at index {i} with window={window}, got {df['rolling_mean'].iloc[i]}"
        )

    # All entries from index window-1 onward must be non-NaN
    for i in range(window - 1, len(df)):
        assert not math.isnan(df["rolling_mean"].iloc[i]), (
            f"Expected non-NaN at index {i} with window={window}, got {df['rolling_mean'].iloc[i]}"
        )


# ---------------------------------------------------------------------------
# Property 4: Signal is 0 or 1 everywhere (Req 5.1, 5.2, 5.3, 5.4)
# ---------------------------------------------------------------------------

@given(close_values=_close_lists, window=_windows)
@settings(max_examples=100)
def test_signal_binary(close_values, window):
    """**Validates: Requirements 5.1, 5.2, 5.3, 5.4**

    For any DataFrame processed by SignalProcessor, every entry in the
    signal column is either 0 or 1 (no nulls, no other values).
    """
    df = pd.DataFrame({"close": close_values}, dtype=float)
    SignalProcessor.process(df, window)

    assert df["signal"].isin([0, 1]).all(), (
        f"Signal column contains values outside {{0, 1}}: {df['signal'].unique().tolist()}"
    )
    assert df["signal"].notna().all(), "Signal column contains NaN values"


# ---------------------------------------------------------------------------
# Property 5: Signal correctness relative to rolling mean (Req 5.2, 5.3, 5.4)
# ---------------------------------------------------------------------------

@given(close_values=_close_lists, window=_windows)
@settings(max_examples=100)
def test_signal_correctness(close_values, window):
    """**Validates: Requirements 5.2, 5.3, 5.4**

    For any row where rolling_mean is non-NaN:
      - signal == 1 iff close > rolling_mean
      - signal == 0 otherwise
    For any row where rolling_mean is NaN, signal == 0.
    """
    df = pd.DataFrame({"close": close_values}, dtype=float)
    SignalProcessor.process(df, window)

    for i in range(len(df)):
        close_val = df["close"].iloc[i]
        rm_val = df["rolling_mean"].iloc[i]
        sig_val = df["signal"].iloc[i]

        if math.isnan(rm_val):
            assert sig_val == 0, (
                f"Row {i}: rolling_mean is NaN but signal={sig_val} (expected 0)"
            )
        elif close_val > rm_val:
            assert sig_val == 1, (
                f"Row {i}: close={close_val} > rolling_mean={rm_val} but signal={sig_val} (expected 1)"
            )
        else:
            assert sig_val == 0, (
                f"Row {i}: close={close_val} <= rolling_mean={rm_val} but signal={sig_val} (expected 0)"
            )


# ---------------------------------------------------------------------------
# Property 6: Metrics output always written (Req 6.7, 8.1, 8.2)
# ---------------------------------------------------------------------------

_ohlcv_header = "timestamp,open,high,low,close,volume_btc,volume_usd\n"


def _build_csv_rows(close_values):
    """Build OHLCV CSV rows from a list of close values."""
    rows = []
    for i, c in enumerate(close_values):
        rows.append(f"2024-01-{(i % 28) + 1:02d},100.0,101.0,99.0,{c},10.0,1000.0")
    return "\n".join(rows) + "\n"


@given(
    close_values=st.lists(_close_floats, min_size=1, max_size=50),
    seed=_seeds,
    window=_windows,
    version=_versions,
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
def test_metrics_always_written(tmp_path, close_values, seed, window, version):
    """**Validates: Requirements 6.7, 8.1, 8.2**

    For any combination of valid inputs, after BatchJob completes, a file
    exists at the --output path and is parseable as JSON with a 'status' key.
    """
    csv_path = tmp_path / "data.csv"
    cfg_path = tmp_path / "config.yaml"
    out_path = tmp_path / "metrics.json"
    log_path = tmp_path / "run.log"

    csv_path.write_text(_ohlcv_header + _build_csv_rows(close_values))
    cfg_path.write_text(f"seed: {seed}\nwindow: {window}\nversion: '{version}'\n")

    argv = [
        "run.py",
        "--input", str(csv_path),
        "--config", str(cfg_path),
        "--output", str(out_path),
        "--log-file", str(log_path),
    ]
    with unittest.mock.patch.object(sys, "argv", argv):
        with pytest.raises(SystemExit):
            run.main()

    # Output file must always exist
    assert out_path.exists(), "metrics.json was not written"

    # Must be valid JSON with a 'status' key
    with open(out_path) as f:
        data = json.load(f)

    assert "status" in data, f"metrics.json missing 'status' key: {data}"


# ---------------------------------------------------------------------------
# Property 7: Reproducibility of signal_rate (Req 9.1, 9.2, 9.3)
# ---------------------------------------------------------------------------

@given(
    close_values=st.lists(_close_floats, min_size=2, max_size=50),
    seed=_seeds,
    window=_windows,
    version=_versions,
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
def test_reproducibility(tmp_path, close_values, seed, window, version):
    """**Validates: Requirements 9.1, 9.2, 9.3**

    For any identical (input CSV, config) pair, running the full pipeline
    twice produces identical value, rows_processed, and signal_rate in
    metrics.json.
    """
    csv_path = tmp_path / "data.csv"
    cfg_path = tmp_path / "config.yaml"
    log_path = tmp_path / "run.log"

    csv_path.write_text(_ohlcv_header + _build_csv_rows(close_values))
    cfg_path.write_text(f"seed: {seed}\nwindow: {window}\nversion: '{version}'\n")

    def run_pipeline(run_id: str) -> dict:
        out_path = tmp_path / f"metrics_{run_id}.json"
        log = tmp_path / f"run_{run_id}.log"
        argv = [
            "run.py",
            "--input", str(csv_path),
            "--config", str(cfg_path),
            "--output", str(out_path),
            "--log-file", str(log),
        ]
        with unittest.mock.patch.object(sys, "argv", argv):
            with pytest.raises(SystemExit) as exc_info:
                run.main()
        assert exc_info.value.code == 0, f"Run {run_id} failed with exit code {exc_info.value.code}"
        with open(out_path) as f:
            return json.load(f)

    metrics1 = run_pipeline("1")
    metrics2 = run_pipeline("2")

    assert metrics1["value"] == metrics2["value"], (
        f"value differs: {metrics1['value']} vs {metrics2['value']}"
    )
    assert metrics1["rows_processed"] == metrics2["rows_processed"], (
        f"rows_processed differs: {metrics1['rows_processed']} vs {metrics2['rows_processed']}"
    )
    # signal_rate is stored in the 'value' field (metric='signal_rate')
    assert metrics1.get("value") == metrics2.get("value"), (
        f"signal_rate (value) differs between runs"
    )


# ---------------------------------------------------------------------------
# Property 8: signal_rate equals mean of signal column (Req 6.3)
# ---------------------------------------------------------------------------

@given(close_values=_close_lists, window=_windows)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_signal_rate_equals_mean(tmp_path, close_values, window):
    """**Validates: Requirements 6.3**

    For any processed DataFrame, metrics.value == round(df["signal"].mean(), 4).
    """
    import time

    df = pd.DataFrame({"close": close_values}, dtype=float)
    SignalProcessor.process(df, window)

    config = Config(seed=42, window=window, version="v1")

    out_path = tmp_path / "metrics.json"
    start_time = time.time()
    MetricsWriter.write_success(str(out_path), df, config, start_time)

    with open(out_path) as f:
        data = json.load(f)

    expected_signal_rate = round(df["signal"].mean(), 4)
    assert data["value"] == expected_signal_rate, (
        f"metrics value={data['value']} != round(signal.mean(), 4)={expected_signal_rate}"
    )
