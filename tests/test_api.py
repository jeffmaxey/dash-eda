"""Tests for eda_core/api.py – EDAAnalyzer class and factory helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from eda_core.api import EDAAnalyzer, from_csv, from_dataframe, from_excel


class TestFromFactories:
    def test_from_dataframe(self, sample_df: pd.DataFrame) -> None:
        analyzer = from_dataframe(sample_df)
        assert isinstance(analyzer, EDAAnalyzer)
        assert analyzer.df.shape == sample_df.shape

    def test_from_csv(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        csv_file = tmp_path / "test.csv"
        sample_df.to_csv(csv_file, index=False)
        analyzer = from_csv(str(csv_file))
        assert isinstance(analyzer, EDAAnalyzer)
        assert analyzer.df.shape[1] == sample_df.shape[1]

    def test_from_excel(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        xl_file = tmp_path / "test.xlsx"
        sample_df.to_excel(xl_file, index=False)
        analyzer = from_excel(str(xl_file))
        assert isinstance(analyzer, EDAAnalyzer)
        assert analyzer.df.shape[1] == sample_df.shape[1]


class TestEDAAnalyzerMethods:
    def test_overview(self, analyzer: EDAAnalyzer) -> None:
        ov = analyzer.overview()
        assert "shape" in ov
        assert "dtypes" in ov
        assert "missing_counts" in ov
        assert "missing_pct" in ov
        assert ov["shape"]["columns"] > 0

    def test_summary_stats(self, analyzer: EDAAnalyzer) -> None:
        ss = analyzer.summary_stats()
        assert "numeric_summary" in ss
        assert "categorical_summary" in ss

    def test_correlation(self, analyzer: EDAAnalyzer) -> None:
        corr = analyzer.correlation()
        assert "columns" in corr
        assert "values" in corr

    def test_column_analysis_numeric(self, analyzer: EDAAnalyzer) -> None:
        result = analyzer.column_analysis("salary")
        assert result["kind"] == "numeric"

    def test_column_analysis_categorical(self, analyzer: EDAAnalyzer) -> None:
        result = analyzer.column_analysis("category")
        assert result["kind"] == "categorical"

    def test_outliers(self, analyzer: EDAAnalyzer) -> None:
        result = analyzer.outliers("salary")
        assert result["method"] == "IQR"
        assert result["outlier_count"] >= 1  # extreme outlier injected in fixture

    def test_to_report(self, analyzer: EDAAnalyzer, tmp_path: Path) -> None:
        output = tmp_path / "report.json"
        analyzer.to_report(str(output))
        assert output.exists()
        data = json.loads(output.read_text())
        assert "overview" in data
        assert "summary_stats" in data
        assert "correlation" in data

    def test_to_report_creates_parent_dirs(
        self, analyzer: EDAAnalyzer, tmp_path: Path
    ) -> None:
        output = tmp_path / "nested" / "dir" / "report.json"
        analyzer.to_report(str(output))
        assert output.exists()
