"""Feature engineering and feature selection utilities.

Provides helpers for:
- Computing feature–target correlations and mutual information.
- Variance-based feature filtering.
- Univariate statistical selection (SelectKBest).
- Tree-based feature importance.
- Creating polynomial and interaction features.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Feature – target correlation / mutual information
# ---------------------------------------------------------------------------


def get_feature_target_stats(
    df: pd.DataFrame,
    target: str,
    problem_type: str = "regression",
) -> dict[str, Any]:
    """Compute correlation / mutual-information between each feature and *target*.

    Parameters
    ----------
    df:
        DataFrame containing features and target.
    target:
        Name of the target column.
    problem_type:
        ``"regression"`` or ``"classification"``.

    Returns
    -------
    dict with keys ``"pearson"``, ``"spearman"``, ``"mutual_info"``
    (each mapping feature name → score).
    """
    feature_cols = [c for c in df.select_dtypes(include="number").columns if c != target]
    if not feature_cols or target not in df.columns:
        return {"pearson": {}, "spearman": {}, "mutual_info": {}}

    clean = df[feature_cols + [target]].dropna()
    if len(clean) < 5:
        return {"pearson": {}, "spearman": {}, "mutual_info": {}}

    X = clean[feature_cols]
    y = clean[target]

    pearson = {col: round(float(clean[col].corr(y, method="pearson")), 4) for col in feature_cols}
    spearman = {col: round(float(clean[col].corr(y, method="spearman")), 4) for col in feature_cols}

    if problem_type == "classification":
        mi_scores = mutual_info_classif(X, y, random_state=42)
    else:
        mi_scores = mutual_info_regression(X, y, random_state=42)

    mutual_info = {col: round(float(s), 4) for col, s in zip(feature_cols, mi_scores)}

    return {"pearson": pearson, "spearman": spearman, "mutual_info": mutual_info}


# ---------------------------------------------------------------------------
# Variance threshold filtering
# ---------------------------------------------------------------------------


def select_by_variance(
    df: pd.DataFrame,
    threshold: float = 0.0,
    exclude: list[str] | None = None,
) -> dict[str, Any]:
    """Return columns that pass the variance threshold.

    Parameters
    ----------
    df:
        Input DataFrame (only numeric columns are evaluated).
    threshold:
        Minimum variance; columns below this are considered low-variance.
    exclude:
        Columns to skip (e.g. the target variable).

    Returns
    -------
    dict with keys ``"selected"``, ``"dropped"``, ``"variances"``.
    """
    exclude = exclude or []
    num_cols = [c for c in df.select_dtypes(include="number").columns if c not in exclude]
    if not num_cols:
        return {"selected": [], "dropped": [], "variances": {}}

    variances = df[num_cols].var().to_dict()
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df[num_cols].fillna(0))
    support = selector.get_support()

    selected = [c for c, s in zip(num_cols, support) if s]
    dropped = [c for c, s in zip(num_cols, support) if not s]
    return {
        "selected": selected,
        "dropped": dropped,
        "variances": {c: round(float(v), 4) for c, v in variances.items()},
    }


# ---------------------------------------------------------------------------
# Univariate statistical selection (SelectKBest)
# ---------------------------------------------------------------------------


def select_k_best(
    df: pd.DataFrame,
    target: str,
    k: int = 10,
    problem_type: str = "regression",
) -> dict[str, Any]:
    """Select the top-*k* features using univariate statistical tests.

    Parameters
    ----------
    df:
        DataFrame containing features and target (numeric only).
    target:
        Target column name.
    k:
        Number of top features to select.
    problem_type:
        ``"regression"`` or ``"classification"``.

    Returns
    -------
    dict with keys ``"selected_features"``, ``"scores"``, ``"pvalues"``.
    """
    feature_cols = [c for c in df.select_dtypes(include="number").columns if c != target]
    if not feature_cols or target not in df.columns:
        return {"selected_features": [], "scores": {}, "pvalues": {}}

    clean = df[feature_cols + [target]].dropna()
    X = clean[feature_cols].values
    y = clean[target].values
    k_actual = min(k, len(feature_cols))

    score_fn = f_classif if problem_type == "classification" else f_regression
    selector = SelectKBest(score_func=score_fn, k=k_actual)
    selector.fit(X, y)
    support = selector.get_support()

    selected = [c for c, s in zip(feature_cols, support) if s]
    scores = {c: round(float(s), 4) for c, s in zip(feature_cols, selector.scores_)}
    pvalues = {c: round(float(p), 6) for c, p in zip(feature_cols, selector.pvalues_)}

    return {"selected_features": selected, "scores": scores, "pvalues": pvalues}


# ---------------------------------------------------------------------------
# Tree-based feature importance
# ---------------------------------------------------------------------------


def get_rf_feature_importance(
    df: pd.DataFrame,
    target: str,
    problem_type: str = "regression",
    n_estimators: int = 100,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compute Random Forest feature importances.

    Parameters
    ----------
    df:
        DataFrame containing numeric features and target.
    target:
        Target column name.
    problem_type:
        ``"regression"`` or ``"classification"``.
    n_estimators:
        Number of trees in the forest.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    dict with keys ``"importances"`` (sorted descending) and ``"selected_top_10"``.
    """
    feature_cols = [c for c in df.select_dtypes(include="number").columns if c != target]
    if not feature_cols or target not in df.columns:
        return {"importances": {}, "selected_top_10": []}

    clean = df[feature_cols + [target]].dropna()
    X = clean[feature_cols].values
    y = clean[target].values

    if problem_type == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)

    model.fit(X, y)
    raw = model.feature_importances_
    importances = dict(
        sorted(
            {col: round(float(imp), 6) for col, imp in zip(feature_cols, raw)}.items(),
            key=lambda x: x[1],
            reverse=True,
        )
    )
    top10 = list(importances.keys())[:10]
    return {"importances": importances, "selected_top_10": top10}


# ---------------------------------------------------------------------------
# Polynomial / interaction feature creation
# ---------------------------------------------------------------------------


def create_polynomial_features(
    df: pd.DataFrame,
    columns: list[str],
    degree: int = 2,
    include_interaction: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Add polynomial (and optionally interaction) features.

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Numeric columns to expand.
    degree:
        Maximum polynomial degree (2 or 3 supported).
    include_interaction:
        Whether to include cross-column interaction terms.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        (augmented DataFrame, list of newly added column names)
    """
    result = df.copy()
    new_cols: list[str] = []

    for col in columns:
        for d in range(2, degree + 1):
            name = f"{col}^{d}"
            result[name] = result[col] ** d
            new_cols.append(name)

    if include_interaction and len(columns) >= 2:
        for i, c1 in enumerate(columns):
            for c2 in columns[i + 1 :]:
                name = f"{c1}*{c2}"
                result[name] = result[c1] * result[c2]
                new_cols.append(name)

    return result, new_cols


def create_interaction_feature(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    operation: str = "multiply",
) -> tuple[pd.DataFrame, str]:
    """Create a single interaction feature between two numeric columns.

    Parameters
    ----------
    operation:
        One of ``"multiply"``, ``"divide"``, ``"add"``, ``"subtract"``.
    """
    result = df.copy()
    ops = {
        "multiply": lambda a, b: a * b,
        "divide": lambda a, b: a / b.replace(0, np.nan),
        "add": lambda a, b: a + b,
        "subtract": lambda a, b: a - b,
    }
    if operation not in ops:
        raise ValueError(f"operation must be one of {list(ops)}")
    name = f"{col1}_{operation}_{col2}"
    result[name] = ops[operation](result[col1], result[col2])
    return result, name
