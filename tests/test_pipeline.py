"""End-to-end pipeline unit tests (Tasks 8.7, 8.8)."""

import json
import sys
import unittest.mock

import pytest

import run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OHLCV_HEADER = "timestamp,open,high,low,close,volume_btc,volume_usd\n"
_OHLCV_ROWS = "".join(
    f"2024-01-{i+1:02d},100.{i},101.{i},99.{i},{100.0 + i},10.0,1000.0\n"
    for i in range(15)
)
_VALID_CSV = _OHLCV_HEADER + _OHLCV_ROWS

_VALID_CONFIG = "seed: 42\nwindow: 5\nversion: v1\n"


def _write_inputs(tmp_path):
    """Write a valid CSV and config YAML into tmp_path; return (csv_path, cfg_path)."""
    csv_path = tmp_path / "data.csv"
    cfg_path = tmp_path / "config.yaml"
    csv_path.write_text(_VALID_CSV)
    cfg_path.write_text(_VALID_CONFIG)
    return str(csv_path), str(cfg_path)


def _run_pipeline(tmp_path, csv_path, cfg_path, run_id=""):
    """Invoke main() with patched sys.argv; return the parsed metrics dict."""
    out_path = str(tmp_path / f"metrics{run_id}.json")
    log_path = str(tmp_path / f"run{run_id}.log")

    argv = [
        "run.py",
        "--input", csv_path,
        "--config", cfg_path,
        "--output", out_path,
        "--log-file", log_path,
    ]
    with unittest.mock.patch.object(sys, "argv", argv):
        with pytest.raises(SystemExit) as exc_info:
            run.main()
    assert exc_info.value.code == 0, f"Pipeline exited with non-zero code: {exc_info.value.code}"

    with open(out_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Task 8.7 — full pipeline success (Req all, 11.7)
# ---------------------------------------------------------------------------

def test_full_pipeline_success(tmp_path):
    """End-to-end run with valid inputs writes metrics.json with status='success'."""
    csv_path, cfg_path = _write_inputs(tmp_path)
    metrics = _run_pipeline(tmp_path, csv_path, cfg_path)

    assert metrics["status"] == "success"
    assert metrics["rows_processed"] == 15
    assert metrics["metric"] == "signal_rate"
    assert "value" in metrics
    assert "latency_ms" in metrics
    assert metrics["seed"] == 42
    assert metrics["version"] == "v1"


# ---------------------------------------------------------------------------
# Task 8.8 — reproducibility with same seed (Req 9, 11.8)
# ---------------------------------------------------------------------------

def test_full_pipeline_reproducible(tmp_path):
    """Two runs with the same seed produce identical 'value' in metrics.json."""
    csv_path, cfg_path = _write_inputs(tmp_path)

    metrics1 = _run_pipeline(tmp_path, csv_path, cfg_path, run_id="1")
    metrics2 = _run_pipeline(tmp_path, csv_path, cfg_path, run_id="2")

    assert metrics1["value"] == metrics2["value"]
    assert metrics1["rows_processed"] == metrics2["rows_processed"]
