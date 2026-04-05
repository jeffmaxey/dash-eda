"""Tests for eda_core/analysis.py."""

from __future__ import annotations

import pandas as pd
import pytest

from eda_core.analysis import (
    detect_outliers,
    get_column_distribution,
    get_correlation_matrix,
    get_missing_heatmap_data,
    get_overview,
    get_summary_stats,
)


class TestGetOverview:
    def test_get_overview_shape(self, sample_df: pd.DataFrame) -> None:
        result = get_overview(sample_df)
        assert result["shape"]["rows"] == len(sample_df)
        assert result["shape"]["columns"] == len(sample_df.columns)

    def test_get_overview_missing(self, sample_df: pd.DataFrame) -> None:
        result = get_overview(sample_df)
        assert "missing_counts" in result
        assert "missing_pct" in result
        assert result["missing_counts"]["age"] > 0
        assert result["missing_pct"]["age"] > 0

    def test_get_overview_duplicate_rows(self, sample_df: pd.DataFrame) -> None:
        result = get_overview(sample_df)
        assert result["duplicate_rows"] >= 3  # we added 3 duplicate rows

    def test_get_overview_memory(self, sample_df: pd.DataFrame) -> None:
        result = get_overview(sample_df)
        assert result["memory_usage_mb"] > 0

    def test_get_overview_dtypes(self, sample_df: pd.DataFrame) -> None:
        result = get_overview(sample_df)
        assert "age" in result["dtypes"]
        assert "category" in result["dtypes"]


class TestGetSummaryStats:
    def test_get_summary_stats_numeric(self, sample_df: pd.DataFrame) -> None:
        result = get_summary_stats(sample_df)
        assert "numeric_summary" in result
        assert "salary" in result["numeric_summary"]
        assert "mean" in result["numeric_summary"]["salary"]

    def test_get_summary_stats_categorical(self, sample_df: pd.DataFrame) -> None:
        result = get_summary_stats(sample_df)
        assert "categorical_summary" in result
        assert "category" in result["categorical_summary"]
        # top-5 value counts
        assert len(result["categorical_summary"]["category"]) <= 5


class TestGetCorrelationMatrix:
    def test_get_correlation_matrix(self, sample_df: pd.DataFrame) -> None:
        result = get_correlation_matrix(sample_df)
        assert "columns" in result
        assert "values" in result
        n = len(result["columns"])
        assert len(result["values"]) == n
        assert all(len(row) == n for row in result["values"])

    def test_correlation_diagonal_is_one(self, sample_df: pd.DataFrame) -> None:
        result = get_correlation_matrix(sample_df)
        for i, row in enumerate(result["values"]):
            assert abs(row[i] - 1.0) < 1e-3

    def test_correlation_single_numeric_column(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = get_correlation_matrix(df)
        assert result["columns"] == ["x"]


class TestGetColumnDistribution:
    def test_get_column_distribution_numeric(self, sample_df: pd.DataFrame) -> None:
        result = get_column_distribution(sample_df, "salary")
        assert result["kind"] == "numeric"
        assert "bins" in result
        assert "counts" in result
        assert len(result["bins"]) == len(result["counts"]) + 1

    def test_get_column_distribution_categorical(self, sample_df: pd.DataFrame) -> None:
        result = get_column_distribution(sample_df, "category")
        assert result["kind"] == "categorical"
        assert "value_counts" in result
        assert isinstance(result["value_counts"], dict)

    def test_get_column_distribution_stats(self, sample_df: pd.DataFrame) -> None:
        result = get_column_distribution(sample_df, "score")
        assert "stats" in result
        assert result["stats"]["mean"] is not None


class TestGetMissingHeatmapData:
    def test_structure(self, sample_df: pd.DataFrame) -> None:
        result = get_missing_heatmap_data(sample_df)
        assert "columns" in result
        assert "is_missing" in result
        assert "row_count" in result
        assert result["row_count"] == len(sample_df)
        assert len(result["columns"]) == len(sample_df.columns)


class TestDetectOutliers:
    def test_detect_outliers(self, sample_df: pd.DataFrame) -> None:
        result = detect_outliers(sample_df, "salary")
        assert result["method"] == "IQR"
        assert "bounds" in result
        assert "outlier_count" in result
        assert result["outlier_count"] >= 1  # we inserted an extreme value

    def test_outlier_pct_range(self, sample_df: pd.DataFrame) -> None:
        result = detect_outliers(sample_df, "score")
        assert 0.0 <= result["outlier_pct"] <= 100.0

    def test_outlier_bounds_ordering(self, sample_df: pd.DataFrame) -> None:
        result = detect_outliers(sample_df, "age")
        assert result["bounds"]["lower"] <= result["bounds"]["upper"]
