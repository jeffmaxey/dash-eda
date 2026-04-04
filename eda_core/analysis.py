"""Core data-analysis functions operating on pandas DataFrames."""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
import pandas as pd


def get_overview(df: pd.DataFrame) -> dict[str, Any]:
    """Return high-level dataset overview.

    Returns a dict with keys:
        shape, dtypes, missing_counts, missing_pct,
        duplicate_rows, memory_usage_mb
    """
    missing_counts = df.isnull().sum().to_dict()
    n_rows = len(df)
    missing_pct = {
        col: round(cnt / n_rows * 100, 2) if n_rows else 0.0
        for col, cnt in missing_counts.items()
    }
    return {
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": {col: int(v) for col, v in missing_counts.items()},
        "missing_pct": missing_pct,
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1_048_576, 4),
    }


def get_summary_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Return numeric describe() and top-5 value_counts for categoricals."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["str", "category", "bool"]
    ).columns.tolist()

    numeric_summary: dict[str, Any] = {}
    if numeric_cols:
        numeric_summary = (
            df[numeric_cols]
            .describe()
            .round(4)
            .to_dict()
        )

    categorical_summary: dict[str, Any] = {}
    for col in categorical_cols:
        vc = df[col].value_counts().head(5)
        categorical_summary[col] = {str(k): int(v) for k, v in vc.items()}

    return {
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
    }


def get_correlation_matrix(df: pd.DataFrame) -> dict[str, Any]:
    """Return correlation matrix for numeric columns as {columns, values}."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty or numeric_df.shape[1] < 2:
        cols = numeric_df.columns.tolist()
        return {
            "columns": cols,
            "values": [[1.0]] if cols else [],
        }
    corr = numeric_df.corr().round(4)
    return {
        "columns": corr.columns.tolist(),
        "values": corr.values.tolist(),
    }


def get_missing_heatmap_data(df: pd.DataFrame) -> dict[str, Any]:
    """Return boolean missingness matrix for heatmap rendering."""
    is_missing = df.isnull()
    return {
        "columns": df.columns.tolist(),
        "is_missing": is_missing.values.tolist(),
        "row_count": len(df),
    }


_MAX_HISTOGRAM_BINS = 30
_MIN_HISTOGRAM_BINS = 5
_HISTOGRAM_BIN_DIVISOR = 10


def get_column_distribution(df: pd.DataFrame, column: str) -> dict[str, Any]:
    """Return distribution data for a single column.

    For numeric columns: histogram bins/counts + descriptive stats.
    For categorical columns: value_counts (all).
    """
    series = df[column]
    dtype = str(series.dtype)

    if pd.api.types.is_numeric_dtype(series):
        clean = series.dropna()
        n_bins = min(
            _MAX_HISTOGRAM_BINS,
            max(_MIN_HISTOGRAM_BINS, len(clean) // _HISTOGRAM_BIN_DIVISOR + 1),
        )
        counts, bin_edges = np.histogram(clean, bins=n_bins)
        stats: dict[str, Any] = {
            "mean": round(float(clean.mean()), 4) if len(clean) else None,
            "median": round(float(clean.median()), 4) if len(clean) else None,
            "std": round(float(clean.std()), 4) if len(clean) else None,
            "min": round(float(clean.min()), 4) if len(clean) else None,
            "max": round(float(clean.max()), 4) if len(clean) else None,
        }
        return {
            "column": column,
            "dtype": dtype,
            "kind": "numeric",
            "stats": stats,
            "bins": bin_edges.tolist(),
            "counts": counts.tolist(),
        }
    else:
        vc = series.value_counts()
        return {
            "column": column,
            "dtype": dtype,
            "kind": "categorical",
            "stats": {"unique": int(series.nunique()), "top": str(vc.index[0]) if len(vc) else None},
            "value_counts": {str(k): int(v) for k, v in vc.items()},
        }


def detect_outliers(df: pd.DataFrame, column: str) -> dict[str, Any]:
    """Detect outliers in a numeric column using the IQR method."""
    series = df[column].dropna()
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_mask = (series < lower) | (series > upper)
    outlier_count = int(outlier_mask.sum())
    return {
        "method": "IQR",
        "column": column,
        "bounds": {"lower": round(lower, 4), "upper": round(upper, 4)},
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "iqr": round(iqr, 4),
        "outlier_count": outlier_count,
        "outlier_pct": round(outlier_count / len(series) * 100, 2) if len(series) else 0.0,
    }


def load_dataframe(path_or_buffer: Any, filename: str) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame, auto-detected by extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in {"xls", "xlsx", "xlsm", "xlsb", "odf", "ods", "odt"}:
        return pd.read_excel(path_or_buffer)
    return pd.read_csv(path_or_buffer)


def parse_upload(contents: str, filename: str) -> pd.DataFrame:
    """Parse a base64-encoded Dash dcc.Upload payload into a DataFrame."""
    _content_type, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    return load_dataframe(io.BytesIO(decoded), filename)
