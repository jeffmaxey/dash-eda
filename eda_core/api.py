"""Public Python API for Dash EDA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from eda_core.analysis import (
    detect_outliers,
    get_bivariate_stats,
    get_column_distribution,
    get_correlation_matrix,
    get_missing_heatmap_data,
    get_multivariate_summary,
    get_overview,
    get_summary_stats,
    load_dataframe,
)
from eda_core.features import (
    create_interaction_feature,
    create_polynomial_features,
    get_feature_target_stats,
    get_rf_feature_importance,
    select_by_variance,
    select_k_best,
)
from eda_core.modeling import (
    compare_models,
    export_model_b64,
    get_available_models,
    import_model_b64,
    predict,
    restore_result,
    serialisable_result,
    train_model,
)
from eda_core.preprocessing import (
    encode_categorical,
    get_preprocessing_summary,
    impute_missing,
    transform_column,
    treat_outliers,
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

    def bivariate(self, col_x: str, col_y: str) -> dict[str, Any]:
        """Return bivariate statistics between two columns."""
        return get_bivariate_stats(self._df, col_x, col_y)

    def multivariate_summary(self) -> dict[str, Any]:
        """Return VIF and eigenvalue-based multivariate summary."""
        return get_multivariate_summary(self._df)

    # ------------------------------------------------------------------
    # Preprocessing methods
    # ------------------------------------------------------------------

    def preprocessing_summary(self) -> dict[str, Any]:
        """Return preprocessing recommendations for the dataset."""
        return get_preprocessing_summary(self._df)

    def impute(
        self,
        columns: list[str] | None = None,
        strategy: str = "mean",
        fill_value: Any = 0,
    ) -> "EDAAnalyzer":
        """Return a new EDAAnalyzer with imputed missing values."""
        df_new, _ = impute_missing(self._df, columns, strategy, fill_value)
        return EDAAnalyzer(df_new)

    def treat_outliers(
        self,
        column: str,
        method: str = "iqr",
        action: str = "cap",
    ) -> "EDAAnalyzer":
        """Return a new EDAAnalyzer with outlier treatment applied."""
        df_new, _ = treat_outliers(self._df, column, method, action)
        return EDAAnalyzer(df_new)

    def transform(
        self,
        column: str,
        transformation: str,
        output_column: str | None = None,
    ) -> "EDAAnalyzer":
        """Return a new EDAAnalyzer with *column* transformed."""
        df_new, _ = transform_column(self._df, column, transformation, output_column)
        return EDAAnalyzer(df_new)

    def encode(
        self,
        column: str,
        method: str = "label",
    ) -> "EDAAnalyzer":
        """Return a new EDAAnalyzer with *column* encoded."""
        df_new, _ = encode_categorical(self._df, column, method)
        return EDAAnalyzer(df_new)

    # ------------------------------------------------------------------
    # Feature methods
    # ------------------------------------------------------------------

    def feature_target_stats(
        self,
        target: str,
        problem_type: str = "regression",
    ) -> dict[str, Any]:
        """Return correlation and mutual-information scores vs *target*."""
        return get_feature_target_stats(self._df, target, problem_type)

    def select_features(
        self,
        target: str,
        k: int = 10,
        problem_type: str = "regression",
    ) -> dict[str, Any]:
        """Return the top-*k* features via SelectKBest."""
        return select_k_best(self._df, target, k, problem_type)

    def feature_importance(
        self,
        target: str,
        problem_type: str = "regression",
    ) -> dict[str, Any]:
        """Return Random Forest feature importances."""
        return get_rf_feature_importance(self._df, target, problem_type)

    # ------------------------------------------------------------------
    # Modeling methods
    # ------------------------------------------------------------------

    def train(
        self,
        target: str,
        model_name: str,
        problem_type: str = "regression",
        test_size: float = 0.2,
        feature_cols: list[str] | None = None,
        scale_features: bool = False,
    ) -> dict[str, Any]:
        """Train a model and return evaluation results."""
        return train_model(
            self._df, target, model_name, problem_type, test_size, feature_cols, scale_features
        )

    def compare(
        self,
        target: str,
        model_names: list[str],
        problem_type: str = "regression",
        test_size: float = 0.2,
    ) -> list[dict[str, Any]]:
        """Train and compare multiple models."""
        return compare_models(self._df, target, model_names, problem_type, test_size)

    def to_report(self, output_path: str) -> None:
        """Save a full JSON report to *output_path*."""
        report: dict[str, Any] = {
            "overview": self.overview(),
            "summary_stats": self.summary_stats(),
            "correlation": self.correlation(),
            "preprocessing_summary": self.preprocessing_summary(),
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
