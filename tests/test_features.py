"""Tests for eda_core/features.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eda_core.features import (
    create_interaction_feature,
    create_polynomial_features,
    get_feature_target_stats,
    get_rf_feature_importance,
    select_by_variance,
    select_k_best,
)


@pytest.fixture
def feature_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 150
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(5, 2, n)
    x3 = rng.uniform(0, 10, n)
    noise = rng.normal(0, 0.5, n)
    y = 2 * x1 - 3 * x2 + noise
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "target": y})


@pytest.fixture
def class_df() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 100
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = (x1 + x2 > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "label": y})


class TestGetFeatureTargetStats:
    def test_regression(self, feature_df: pd.DataFrame) -> None:
        result = get_feature_target_stats(feature_df, "target", "regression")
        assert "pearson" in result
        assert "spearman" in result
        assert "mutual_info" in result
        assert "x1" in result["pearson"]
        assert "x2" in result["pearson"]

    def test_classification(self, class_df: pd.DataFrame) -> None:
        result = get_feature_target_stats(class_df, "label", "classification")
        assert "mutual_info" in result

    def test_missing_target_returns_empty(self, feature_df: pd.DataFrame) -> None:
        result = get_feature_target_stats(feature_df, "nonexistent", "regression")
        assert result["pearson"] == {}


class TestSelectByVariance:
    def test_zero_variance_dropped(self) -> None:
        df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
        result = select_by_variance(df, threshold=0.0)
        assert "b" in result["selected"]
        # "a" has zero variance and should appear in dropped
        assert "a" in result["dropped"]

    def test_exclude_target(self, feature_df: pd.DataFrame) -> None:
        result = select_by_variance(feature_df, exclude=["target"])
        assert "target" not in result["selected"]
        assert "target" not in result["dropped"]

    def test_variances_present(self, feature_df: pd.DataFrame) -> None:
        result = select_by_variance(feature_df)
        assert "x1" in result["variances"]


class TestSelectKBest:
    def test_regression_top_k(self, feature_df: pd.DataFrame) -> None:
        result = select_k_best(feature_df, "target", k=2, problem_type="regression")
        assert len(result["selected_features"]) == 2
        assert "scores" in result
        assert "pvalues" in result

    def test_classification_top_k(self, class_df: pd.DataFrame) -> None:
        result = select_k_best(class_df, "label", k=1, problem_type="classification")
        assert len(result["selected_features"]) == 1

    def test_empty_when_no_target(self, feature_df: pd.DataFrame) -> None:
        result = select_k_best(feature_df, "missing_col", k=2)
        assert result["selected_features"] == []

    def test_k_capped_at_n_features(self, feature_df: pd.DataFrame) -> None:
        result = select_k_best(feature_df, "target", k=100)
        assert len(result["selected_features"]) <= 3  # 3 feature cols


class TestGetRFFeatureImportance:
    def test_regression_importances(self, feature_df: pd.DataFrame) -> None:
        result = get_rf_feature_importance(feature_df, "target", "regression", n_estimators=20)
        assert "importances" in result
        assert len(result["importances"]) == 3
        total = sum(result["importances"].values())
        assert abs(total - 1.0) < 0.01

    def test_classification_importances(self, class_df: pd.DataFrame) -> None:
        result = get_rf_feature_importance(class_df, "label", "classification", n_estimators=20)
        assert len(result["importances"]) == 2

    def test_sorted_descending(self, feature_df: pd.DataFrame) -> None:
        result = get_rf_feature_importance(feature_df, "target", n_estimators=20)
        vals = list(result["importances"].values())
        assert vals == sorted(vals, reverse=True)

    def test_top_10_key(self, feature_df: pd.DataFrame) -> None:
        result = get_rf_feature_importance(feature_df, "target", n_estimators=20)
        assert "selected_top_10" in result


class TestCreatePolynomialFeatures:
    def test_degree_2(self, feature_df: pd.DataFrame) -> None:
        df_out, new_cols = create_polynomial_features(feature_df, ["x1", "x2"], degree=2)
        assert "x1^2" in new_cols
        assert "x2^2" in new_cols
        assert "x1*x2" in new_cols

    def test_degree_3(self, feature_df: pd.DataFrame) -> None:
        df_out, new_cols = create_polynomial_features(
            feature_df, ["x1"], degree=3, include_interaction=False
        )
        assert "x1^3" in new_cols

    def test_no_interaction(self, feature_df: pd.DataFrame) -> None:
        _, new_cols = create_polynomial_features(
            feature_df, ["x1", "x2"], degree=2, include_interaction=False
        )
        assert "x1*x2" not in new_cols


class TestCreateInteractionFeature:
    def test_multiply(self, feature_df: pd.DataFrame) -> None:
        df_out, name = create_interaction_feature(feature_df, "x1", "x2", "multiply")
        assert name == "x1_multiply_x2"
        assert name in df_out.columns
        expected = feature_df["x1"] * feature_df["x2"]
        pd.testing.assert_series_equal(df_out[name], expected, check_names=False)

    def test_add(self, feature_df: pd.DataFrame) -> None:
        df_out, name = create_interaction_feature(feature_df, "x1", "x2", "add")
        assert name == "x1_add_x2"

    def test_invalid_operation_raises(self, feature_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError):
            create_interaction_feature(feature_df, "x1", "x2", "bad_op")
