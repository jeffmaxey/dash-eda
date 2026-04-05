"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eda_core.api import EDAAnalyzer


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """DataFrame with mixed dtypes, some NaN values, and an outlier."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 80, size=n).astype(float),
            "salary": rng.normal(55_000, 15_000, size=n),
            "score": rng.uniform(0, 100, size=n),
            "category": rng.choice(["A", "B", "C", "D"], size=n),
            "flag": rng.choice([True, False], size=n),
            "city": rng.choice(["London", "Paris", "Berlin", "Madrid"], size=n),
        }
    )
    # Introduce NaN values
    df.loc[rng.choice(df.index, 10, replace=False), "age"] = np.nan
    df.loc[rng.choice(df.index, 5, replace=False), "category"] = np.nan
    # Introduce an extreme outlier
    df.loc[0, "salary"] = 1_000_000.0
    # Introduce duplicate rows
    df = pd.concat([df, df.iloc[:3]], ignore_index=True)
    return df


@pytest.fixture
def analyzer(sample_df: pd.DataFrame) -> EDAAnalyzer:
    """EDAAnalyzer wrapping sample_df."""
    return EDAAnalyzer(sample_df)
