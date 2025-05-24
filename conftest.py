"""Test set-up and fixtures code."""

from datetime import datetime
from typing import Any

import polars as pl
import pytest


@pytest.fixture(autouse=True)
def __MEDS_visualizations_setup_doctest_namespace(doctest_namespace: dict[str, Any]) -> None:
    doctest_namespace.update(
        {
            "pl": pl,
            "datetime": datetime,
        }
    )
