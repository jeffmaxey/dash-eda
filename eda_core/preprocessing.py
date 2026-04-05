"""Data preprocessing utilities for EDA and ML pipelines.

Covers missing-value imputation, outlier treatment, numeric
transformations, and categorical encoding.  All functions return
a *new* DataFrame and, where applicable, a summary dict so the
dashboard can display what changed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Missing-value imputation
# ---------------------------------------------------------------------------

IMPUTE_STRATEGIES = ("mean", "median", "mode", "constant", "drop")


def impute_missing(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    strategy: str = "mean",
    fill_value: Any = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Impute missing values in *columns* using *strategy*.

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Columns to impute.  ``None`` means all columns with any missing.
    strategy:
        One of ``"mean"``, ``"median"``, ``"mode"``, ``"constant"``, ``"drop"``.
    fill_value:
        Value used when *strategy* is ``"constant"``.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (transformed DataFrame, summary of changes)
    """
    if strategy not in IMPUTE_STRATEGIES:
        raise ValueError(f"strategy must be one of {IMPUTE_STRATEGIES}")

    target_cols = columns if columns is not None else df.columns[df.isnull().any()].tolist()
    result = df.copy()
    summary: dict[str, Any] = {"strategy": strategy, "columns": {}}

    if strategy == "drop":
        before = len(result)
        result = result.dropna(subset=target_cols)
        summary["rows_dropped"] = before - len(result)
        return result, summary

    for col in target_cols:
        missing_before = int(result[col].isnull().sum())
        if missing_before == 0:
            continue
        if strategy == "mean":
            if pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].fillna(result[col].mean())
        elif strategy == "median":
            if pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].fillna(result[col].median())
        elif strategy == "mode":
            mode_val = result[col].mode()
            if len(mode_val):
                result[col] = result[col].fillna(mode_val.iloc[0])
        elif strategy == "constant":
            result[col] = result[col].fillna(fill_value)
        summary["columns"][col] = {
            "missing_before": missing_before,
            "missing_after": int(result[col].isnull().sum()),
        }

    return result, summary


# ---------------------------------------------------------------------------
# Outlier treatment
# ---------------------------------------------------------------------------

OUTLIER_METHODS = ("iqr", "zscore")
OUTLIER_ACTIONS = ("cap", "remove", "flag")


def treat_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
    action: str = "cap",
    z_threshold: float = 3.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Detect and treat outliers in a numeric column.

    Parameters
    ----------
    df:
        Input DataFrame.
    column:
        Numeric column to process.
    method:
        Detection method: ``"iqr"`` or ``"zscore"``.
    action:
        What to do with outliers: ``"cap"`` (Winsorise), ``"remove"``,
        or ``"flag"`` (add boolean column).
    z_threshold:
        Z-score threshold used only when *method* is ``"zscore"``.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (transformed DataFrame, summary)
    """
    if method not in OUTLIER_METHODS:
        raise ValueError(f"method must be one of {OUTLIER_METHODS}")
    if action not in OUTLIER_ACTIONS:
        raise ValueError(f"action must be one of {OUTLIER_ACTIONS}")

    result = df.copy()
    series = result[column].dropna()

    if method == "iqr":
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
    else:  # zscore
        mean = float(series.mean())
        std = float(series.std())
        lower = mean - z_threshold * std
        upper = mean + z_threshold * std

    outlier_mask = (result[column] < lower) | (result[column] > upper)
    outlier_count = int(outlier_mask.sum())

    if action == "cap":
        result[column] = result[column].clip(lower=lower, upper=upper)
    elif action == "remove":
        result = result[~outlier_mask].reset_index(drop=True)
    elif action == "flag":
        result[f"{column}_outlier"] = outlier_mask.astype(int)

    return result, {
        "column": column,
        "method": method,
        "action": action,
        "bounds": {"lower": round(lower, 4), "upper": round(upper, 4)},
        "outliers_found": outlier_count,
        "rows_after": len(result),
    }


# ---------------------------------------------------------------------------
# Column transformations
# ---------------------------------------------------------------------------

TRANSFORMATIONS = (
    "log1p",
    "sqrt",
    "square",
    "standardize",
    "normalize",
    "boxcox",
    "yeojohnson",
)


def transform_column(
    df: pd.DataFrame,
    column: str,
    transformation: str,
    output_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply a numeric transformation to *column*.

    Parameters
    ----------
    df:
        Input DataFrame.
    column:
        Numeric column to transform.
    transformation:
        One of the values in :data:`TRANSFORMATIONS`.
    output_column:
        Name for the new/replaced column.  Defaults to
        ``f"{column}_{transformation}"``.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (transformed DataFrame, summary)
    """
    if transformation not in TRANSFORMATIONS:
        raise ValueError(f"transformation must be one of {TRANSFORMATIONS}")

    result = df.copy()
    out_col = output_column or f"{column}_{transformation}"
    series = result[column].dropna()

    skew_before = round(float(series.skew()), 4)

    if transformation == "log1p":
        result[out_col] = np.log1p(result[column])
    elif transformation == "sqrt":
        result[out_col] = np.sqrt(result[column].clip(lower=0))
    elif transformation == "square":
        result[out_col] = result[column] ** 2
    elif transformation == "standardize":
        mean = float(series.mean())
        std = float(series.std())
        result[out_col] = (result[column] - mean) / (std if std else 1.0)
    elif transformation == "normalize":
        vmin = float(series.min())
        vmax = float(series.max())
        denom = vmax - vmin if vmax != vmin else 1.0
        result[out_col] = (result[column] - vmin) / denom
    elif transformation == "boxcox":
        clean = series[series > 0]
        transformed, _ = scipy_stats.boxcox(clean)
        result[out_col] = np.nan
        result.loc[clean.index, out_col] = transformed
    elif transformation == "yeojohnson":
        transformed, _ = scipy_stats.yeojohnson(series.values)
        result.loc[series.index, out_col] = transformed

    new_series = result[out_col].dropna()
    skew_after = round(float(new_series.skew()), 4) if len(new_series) else None

    return result, {
        "column": column,
        "output_column": out_col,
        "transformation": transformation,
        "skew_before": skew_before,
        "skew_after": skew_after,
    }


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------

ENCODING_METHODS = ("label", "onehot", "ordinal", "frequency")


def encode_categorical(
    df: pd.DataFrame,
    column: str,
    method: str = "label",
    ordinal_order: list[str] | None = None,
    drop_original: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Encode a categorical column.

    Parameters
    ----------
    df:
        Input DataFrame.
    column:
        Categorical column to encode.
    method:
        One of ``"label"``, ``"onehot"``, ``"ordinal"``, ``"frequency"``.
    ordinal_order:
        Explicit category order for *ordinal* encoding.
    drop_original:
        Whether to drop the original column (except for ``"onehot"``
        which always drops it when ``drop_original`` is ``True``).

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (transformed DataFrame, summary)
    """
    if method not in ENCODING_METHODS:
        raise ValueError(f"method must be one of {ENCODING_METHODS}")

    result = df.copy()
    new_cols: list[str] = []

    if method == "label":
        cats = result[column].astype("category")
        result[f"{column}_encoded"] = cats.cat.codes
        new_cols = [f"{column}_encoded"]
    elif method == "onehot":
        dummies = pd.get_dummies(result[column], prefix=column, dtype=int)
        result = pd.concat([result, dummies], axis=1)
        new_cols = dummies.columns.tolist()
        if drop_original:
            result = result.drop(columns=[column])
    elif method == "ordinal":
        order = ordinal_order or sorted(result[column].dropna().unique().tolist())
        mapping = {v: i for i, v in enumerate(order)}
        result[f"{column}_ordinal"] = result[column].map(mapping)
        new_cols = [f"{column}_ordinal"]
    elif method == "frequency":
        freq = result[column].value_counts(normalize=True)
        result[f"{column}_freq"] = result[column].map(freq)
        new_cols = [f"{column}_freq"]

    if drop_original and method != "onehot" and column in result.columns:
        result = result.drop(columns=[column])

    return result, {
        "column": column,
        "method": method,
        "new_columns": new_cols,
        "unique_values": int(df[column].nunique()),
    }


# ---------------------------------------------------------------------------
# Preprocessing summary
# ---------------------------------------------------------------------------


def get_preprocessing_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Return a quick summary of data quality issues useful for preprocessing.

    Includes per-column missingness, skewness (numeric), and cardinality
    (categorical).
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    n = len(df)

    missing: dict[str, Any] = {}
    for col in df.columns:
        cnt = int(df[col].isnull().sum())
        if cnt:
            missing[col] = {"count": cnt, "pct": round(cnt / n * 100, 2) if n else 0.0}

    skewness: dict[str, float] = {}
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) > 2:
            skewness[col] = round(float(s.skew()), 4)

    cardinality: dict[str, int] = {col: int(df[col].nunique()) for col in cat_cols}

    return {
        "n_rows": n,
        "n_cols": len(df.columns),
        "numeric_cols": numeric_cols,
        "categorical_cols": cat_cols,
        "missing": missing,
        "skewness": skewness,
        "cardinality": cardinality,
    }
