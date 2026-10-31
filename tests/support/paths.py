"""Repository locations and per-module test data lookup."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
EXAMPLES = REPO_ROOT / "examples"


def data_file(test_file: str, name: str) -> Path:
    """Return ``data/<name>.yaml`` next to a test module, failing clearly if it is missing."""
    path = Path(test_file).parent / "data" / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"test data not found: {path.relative_to(REPO_ROOT)}")
    return path


def cases(test_file: str, name: str) -> list[Any]:
    """Load a tabular data file as one ``pytest.param`` per row, identified by its ``id``."""
    rows = yaml.safe_load(data_file(test_file, name).read_text(encoding="utf-8"))
    return [pytest.param(row, id=row["id"]) for row in rows]
