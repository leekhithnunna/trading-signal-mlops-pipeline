"""Unit tests for ConfigLoader (Tasks 8.1, 8.2)."""

import pytest

from src.config import ConfigLoader


# ---------------------------------------------------------------------------
# Task 8.1 — valid YAML returns Config with correct fields (Req 2.1, 11.1)
# ---------------------------------------------------------------------------

def test_config_loader_valid(tmp_path):
    """A valid YAML file with seed, window, and version returns a Config with matching fields."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("seed: 42\nwindow: 5\nversion: v1\n")

    config = ConfigLoader.load(str(cfg_file))

    assert config.seed == 42
    assert config.window == 5
    assert config.version == "v1"


# ---------------------------------------------------------------------------
# Task 8.2 — missing field raises ValueError (Req 2.2–2.4, 11.2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("missing_field,yaml_content", [
    ("seed",    "window: 5\nversion: v1\n"),
    ("window",  "seed: 42\nversion: v1\n"),
    ("version", "seed: 42\nwindow: 5\n"),
])
def test_config_loader_missing_field(tmp_path, missing_field, yaml_content):
    """A config YAML missing any one required field raises a ValueError naming that field."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml_content)

    with pytest.raises(ValueError, match=missing_field):
        ConfigLoader.load(str(cfg_file))
