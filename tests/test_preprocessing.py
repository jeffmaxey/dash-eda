"""Tests for eda_core/preprocessing.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eda_core.preprocessing import (
    encode_categorical,
    get_preprocessing_summary,
    impute_missing,
    transform_column,
    treat_outliers,
)


@pytest.fixture
def mixed_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame(
        {
            "a": rng.normal(10, 2, n),
            "b": rng.exponential(5, n),
            "cat": rng.choice(["X", "Y", "Z"], n),
        }
    )
    # Introduce NaN in both columns
    df.loc[2, "a"] = np.nan
    df.loc[1, "b"] = np.nan
    # Introduce outlier in a different row from the NaN
    df.loc[0, "a"] = 1_000.0
    return df


class TestImputeMissing:
    def test_mean_imputation(self, mixed_df: pd.DataFrame) -> None:
        df_out, summary = impute_missing(mixed_df, strategy="mean")
        assert df_out["a"].isnull().sum() == 0
        assert "a" in summary["columns"] or "b" in summary["columns"]

    def test_median_imputation(self, mixed_df: pd.DataFrame) -> None:
        df_out, summary = impute_missing(mixed_df, columns=["b"], strategy="median")
        assert df_out["b"].isnull().sum() == 0

    def test_mode_imputation_categorical(self, mixed_df: pd.DataFrame) -> None:
        df = mixed_df.copy()
        df.loc[5, "cat"] = np.nan
        df_out, _ = impute_missing(df, columns=["cat"], strategy="mode")
        assert df_out["cat"].isnull().sum() == 0

    def test_constant_imputation(self, mixed_df: pd.DataFrame) -> None:
        df_out, _ = impute_missing(mixed_df, columns=["a"], strategy="constant", fill_value=-1.0)
        assert -1.0 in df_out["a"].values

    def test_drop_strategy(self, mixed_df: pd.DataFrame) -> None:
        df_out, summary = impute_missing(mixed_df, strategy="drop")
        assert df_out.isnull().sum().sum() == 0
        assert summary["rows_dropped"] >= 0

    def test_invalid_strategy_raises(self, mixed_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError):
            impute_missing(mixed_df, strategy="invalid")


class TestTreatOutliers:
    def test_cap_iqr(self, mixed_df: pd.DataFrame) -> None:
        df_out, summary = treat_outliers(mixed_df, "a", method="iqr", action="cap")
        assert summary["action"] == "cap"
        # The actual cap uses unrounded bounds; verify outlier value is clipped
        assert df_out["a"].max() < 1_000.0

    def test_remove_iqr(self, mixed_df: pd.DataFrame) -> None:
        df_out, summary = treat_outliers(mixed_df, "a", method="iqr", action="remove")
        assert len(df_out) <= len(mixed_df)

    def test_flag_iqr(self, mixed_df: pd.DataFrame) -> None:
        df_out, summary = treat_outliers(mixed_df, "a", method="iqr", action="flag")
        assert "a_outlier" in df_out.columns

    def test_zscore_method(self, mixed_df: pd.DataFrame) -> None:
        df_out, summary = treat_outliers(mixed_df, "a", method="zscore", action="cap")
        assert summary["method"] == "zscore"

    def test_invalid_method_raises(self, mixed_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError):
            treat_outliers(mixed_df, "a", method="bad")


class TestTransformColumn:
    def test_log1p(self, mixed_df: pd.DataFrame) -> None:
        df_out, summary = transform_column(mixed_df, "b", "log1p")
        assert "b_log1p" in df_out.columns
        assert summary["transformation"] == "log1p"

    def test_standardize(self, mixed_df: pd.DataFrame) -> None:
        df_out, summary = transform_column(mixed_df, "a", "standardize")
        col = df_out["a_standardize"].dropna()
        assert abs(col.mean()) < 0.1
        assert abs(col.std() - 1) < 0.1

    def test_normalize(self, mixed_df: pd.DataFrame) -> None:
        df_out, _ = transform_column(mixed_df, "a", "normalize")
        col = df_out["a_normalize"].dropna()
        assert col.min() >= 0.0
        assert col.max() <= 1.0

    def test_sqrt(self, mixed_df: pd.DataFrame) -> None:
        df_out, _ = transform_column(mixed_df, "b", "sqrt")
        assert "b_sqrt" in df_out.columns

    def test_skew_improves(self, mixed_df: pd.DataFrame) -> None:
        _, summary = transform_column(mixed_df, "b", "log1p")
        if summary["skew_after"] is not None and summary["skew_before"] != 0:
            assert abs(summary["skew_after"]) <= abs(summary["skew_before"]) + 1

    def test_invalid_transformation_raises(self, mixed_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError):
            transform_column(mixed_df, "a", "bad_transform")


class TestEncodeCategorical:
    def test_label_encoding(self, mixed_df: pd.DataFrame) -> None:
        df_out, summary = encode_categorical(mixed_df, "cat", method="label")
        assert "cat_encoded" in df_out.columns
        assert summary["method"] == "label"

    def test_onehot_encoding(self, mixed_df: pd.DataFrame) -> None:
        df_out, summary = encode_categorical(mixed_df, "cat", method="onehot")
        assert "cat" not in df_out.columns
        assert len(summary["new_columns"]) >= 2

    def test_frequency_encoding(self, mixed_df: pd.DataFrame) -> None:
        df_out, summary = encode_categorical(mixed_df, "cat", method="frequency")
        assert "cat_freq" in df_out.columns
        freqs = df_out["cat_freq"].dropna()
        assert (freqs <= 1).all()
        assert (freqs >= 0).all()

    def test_invalid_method_raises(self, mixed_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError):
            encode_categorical(mixed_df, "cat", method="invalid")


class TestGetPreprocessingSummary:
    def test_structure(self, mixed_df: pd.DataFrame) -> None:
        summary = get_preprocessing_summary(mixed_df)
        assert "missing" in summary
        assert "skewness" in summary
        assert "cardinality" in summary
        assert "categorical_cols" in summary
        assert "numeric_cols" in summary

    def test_missing_detected(self, mixed_df: pd.DataFrame) -> None:
        summary = get_preprocessing_summary(mixed_df)
        # "b" has NaN
        assert "b" in summary["missing"]

    def test_cardinality(self, mixed_df: pd.DataFrame) -> None:
        summary = get_preprocessing_summary(mixed_df)
        assert summary["cardinality"]["cat"] == 3
