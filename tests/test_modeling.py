"""Tests for eda_core/modeling.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eda_core.modeling import (
    compare_models,
    export_model_b64,
    export_model_bytes,
    get_available_models,
    import_model_b64,
    import_model_bytes,
    predict,
    restore_result,
    serialisable_result,
    train_model,
)


@pytest.fixture
def regression_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(5, 2, n)
    y = 2 * x1 - 1.5 * x2 + rng.normal(0, 0.3, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


@pytest.fixture
def classification_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    label = (x1 + x2 > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "label": label})


class TestGetAvailableModels:
    def test_regression_models_present(self) -> None:
        models = get_available_models("regression")
        assert "Linear Regression" in models
        assert "Random Forest" in models

    def test_classification_models_present(self) -> None:
        models = get_available_models("classification")
        assert "Logistic Regression" in models
        assert "Random Forest" in models


class TestTrainModel:
    def test_regression_basic(self, regression_df: pd.DataFrame) -> None:
        result = train_model(regression_df, "y", "Linear Regression", "regression")
        assert result["problem_type"] == "regression"
        assert "metrics" in result
        m = result["metrics"]
        assert "r2" in m
        assert "mae" in m
        assert m["r2"] > 0.5  # should be high for this linear data

    def test_classification_basic(self, classification_df: pd.DataFrame) -> None:
        result = train_model(
            classification_df, "label", "Logistic Regression", "classification"
        )
        assert "metrics" in result
        assert "accuracy" in result["metrics"]
        assert result["metrics"]["accuracy"] > 0.7

    def test_random_forest_regression(self, regression_df: pd.DataFrame) -> None:
        result = train_model(regression_df, "y", "Random Forest", "regression")
        assert result["metrics"]["r2"] > 0.5

    def test_feature_importance_present(self, regression_df: pd.DataFrame) -> None:
        result = train_model(regression_df, "y", "Random Forest", "regression")
        assert len(result["feature_importance"]) > 0

    def test_cv_scores_present(self, regression_df: pd.DataFrame) -> None:
        result = train_model(regression_df, "y", "Linear Regression", "regression", cv_folds=3)
        assert result["cv_scores"]["mean"] is not None

    def test_scale_features(self, regression_df: pd.DataFrame) -> None:
        result = train_model(
            regression_df, "y", "Linear Regression", "regression", scale_features=True
        )
        assert result["_scaler"] is not None

    def test_samples_add_up(self, regression_df: pd.DataFrame) -> None:
        result = train_model(regression_df, "y", "Linear Regression", "regression", test_size=0.2)
        total = result["train_samples"] + result["test_samples"]
        assert total == len(regression_df)

    def test_invalid_model_raises(self, regression_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError):
            train_model(regression_df, "y", "No Such Model", "regression")

    def test_confusion_matrix_in_classification(self, classification_df: pd.DataFrame) -> None:
        result = train_model(
            classification_df, "label", "Random Forest", "classification"
        )
        assert "confusion_matrix" in result["metrics"]

    def test_roc_curve_binary_classification(self, classification_df: pd.DataFrame) -> None:
        result = train_model(
            classification_df, "label", "Logistic Regression", "classification"
        )
        assert "roc_auc" in result["metrics"]
        assert "roc_curve" in result["metrics"]


class TestCompareModels:
    def test_compare_regression(self, regression_df: pd.DataFrame) -> None:
        models = ["Linear Regression", "Random Forest"]
        results = compare_models(regression_df, "y", models, "regression")
        assert len(results) == 2
        # Should be sorted by r2 descending
        r2_scores = [r.get("metrics", {}).get("r2", -999) for r in results]
        assert r2_scores[0] >= r2_scores[1]

    def test_error_model_included(self, regression_df: pd.DataFrame) -> None:
        results = compare_models(regression_df, "y", ["Linear Regression", "No Model"], "regression")
        model_names = [r["model_name"] for r in results]
        assert "No Model" in model_names
        error_result = next(r for r in results if r["model_name"] == "No Model")
        assert "error" in error_result


class TestPredict:
    def test_regression_predict(self, regression_df: pd.DataFrame) -> None:
        result = train_model(regression_df, "y", "Linear Regression", "regression")
        pred_out = predict(result, regression_df)
        assert "predictions" in pred_out
        assert len(pred_out["predictions"]) == len(regression_df)

    def test_classification_predict_labels(self, classification_df: pd.DataFrame) -> None:
        result = train_model(
            classification_df, "label", "Logistic Regression", "classification"
        )
        pred_out = predict(result, classification_df)
        assert "predictions" in pred_out
        if hasattr(result.get("_le"), "classes_"):
            assert "predictions_labels" in pred_out

    def test_predict_probabilities(self, classification_df: pd.DataFrame) -> None:
        result = train_model(
            classification_df, "label", "Random Forest", "classification"
        )
        pred_out = predict(result, classification_df)
        assert "probabilities" in pred_out


class TestSerialization:
    def test_serialisable_round_trip(self, regression_df: pd.DataFrame) -> None:
        result = train_model(regression_df, "y", "Linear Regression", "regression")
        clean = serialisable_result(result)
        assert "_model" not in clean
        assert "model_b64" in clean
        restored = restore_result(clean)
        assert "_model" in restored

    def test_export_bytes(self, regression_df: pd.DataFrame) -> None:
        result = train_model(regression_df, "y", "Linear Regression", "regression")
        data = export_model_bytes(result)
        assert isinstance(data, bytes)
        assert len(data) > 100

    def test_import_bytes_roundtrip(self, regression_df: pd.DataFrame) -> None:
        result = train_model(regression_df, "y", "Linear Regression", "regression")
        data = export_model_bytes(result)
        payload = import_model_bytes(data)
        assert "model" in payload
        assert "feature_cols" in payload

    def test_b64_roundtrip(self, regression_df: pd.DataFrame) -> None:
        result = train_model(regression_df, "y", "Linear Regression", "regression")
        b64 = export_model_b64(result)
        payload = import_model_b64(b64)
        assert "model" in payload

    def test_predictions_after_roundtrip(self, regression_df: pd.DataFrame) -> None:
        result = train_model(regression_df, "y", "Random Forest", "regression")
        clean = serialisable_result(result)
        restored = restore_result(clean)
        pred_out = predict(restored, regression_df)
        assert len(pred_out["predictions"]) == len(regression_df)
