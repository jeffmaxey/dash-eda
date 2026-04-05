"""Public Python API for Dash EDA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from eda_core.analysis import (
    detect_outliers,
    get_column_distribution,
    get_correlation_matrix,
    get_overview,
    get_summary_stats,
    load_dataframe,
)


class EDAAnalyzer:
    """High-level EDA helper that wraps a pandas DataFrame."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------

    def overview(self) -> dict[str, Any]:
        """Return high-level dataset overview."""
        return get_overview(self._df)

    def summary_stats(self) -> dict[str, Any]:
        """Return numeric and categorical summary statistics."""
        return get_summary_stats(self._df)

    def correlation(self) -> dict[str, Any]:
        """Return the Pearson correlation matrix for numeric columns."""
        return get_correlation_matrix(self._df)

    def column_analysis(self, column: str) -> dict[str, Any]:
        """Return distribution data for *column*."""
        return get_column_distribution(self._df, column)

    def outliers(self, column: str) -> dict[str, Any]:
        """Return IQR-based outlier statistics for *column*."""
        return detect_outliers(self._df, column)

    def to_report(self, output_path: str) -> None:
        """Save a full JSON report to *output_path*."""
        report: dict[str, Any] = {
            "overview": self.overview(),
            "summary_stats": self.summary_stats(),
            "correlation": self.correlation(),
        }
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def __repr__(self) -> str:  # pragma: no cover
        return f"EDAAnalyzer(shape={self._df.shape})"


# ---------------------------------------------------------------------------
# Module-level factory helpers
# ---------------------------------------------------------------------------


def from_csv(path: str) -> EDAAnalyzer:
    """Create an EDAAnalyzer from a CSV file."""
    return EDAAnalyzer(load_dataframe(path, path))


def from_excel(path: str) -> EDAAnalyzer:
    """Create an EDAAnalyzer from an Excel file."""
    df = pd.read_excel(path)
    return EDAAnalyzer(df)


def from_dataframe(df: pd.DataFrame) -> EDAAnalyzer:
    """Create an EDAAnalyzer from an existing DataFrame."""
    return EDAAnalyzer(df)
