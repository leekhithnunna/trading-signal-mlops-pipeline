import os
import pytest

from src.validator import DataValidator


def test_data_validator_file_not_found(tmp_path):
    """Task 8.3 / Req 3.1 — Raises FileNotFoundError when path does not exist."""
    missing = str(tmp_path / "nonexistent.csv")
    with pytest.raises(FileNotFoundError):
        DataValidator.validate(missing)


def test_csv_parse_failure(tmp_path):
    """Task 3.3 — Raises ValueError on CSV parse failure (malformed file)."""
    bad_csv = tmp_path / "bad.csv"
    # Unclosed quote causes a pandas ParserError
    bad_csv.write_text('close,open\n"unclosed,1\n2,3\n')
    with pytest.raises(ValueError, match="CSV parse error"):
        DataValidator.validate(str(bad_csv))


def test_data_validator_empty_csv(tmp_path):
    """Task 8.4 / Req 3.3 — Raises ValueError when DataFrame has zero rows (header-only CSV)."""
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("close,open,high,low\n")
    with pytest.raises(ValueError, match="Dataset is empty"):
        DataValidator.validate(str(empty_csv))


def test_data_validator_missing_close(tmp_path):
    """Task 8.5 / Req 3.4 — Raises ValueError('Missing required column: close') when close absent."""
    no_close = tmp_path / "no_close.csv"
    no_close.write_text("open,high,low,volume\n1,2,3,4\n5,6,7,8\n")
    with pytest.raises(ValueError, match="Missing required column: close"):
        DataValidator.validate(str(no_close))


def test_valid_csv_returns_dataframe(tmp_path):
    """Task 3.6 — Returns parsed DataFrame on success."""
    valid_csv = tmp_path / "valid.csv"
    valid_csv.write_text("close,open,high,low\n100.0,99.0,101.0,98.0\n200.0,199.0,201.0,198.0\n")
    df = DataValidator.validate(str(valid_csv))
    assert len(df) == 2
    assert "close" in df.columns
    assert list(df["close"]) == [100.0, 200.0]
