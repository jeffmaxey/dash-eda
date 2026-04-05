"""Machine-learning model construction, evaluation, comparison, and I/O.

Supports both *classification* and *regression* problem types using
scikit-learn estimators.  All public functions operate on plain
DataFrames and return JSON-serialisable dictionaries.
"""

from __future__ import annotations

import io
import base64
import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_CLASSIFIERS: dict[str, Any] = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True, random_state=42),
}

_REGRESSORS: dict[str, Any] = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(random_state=42),
    "Lasso": Lasso(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "KNN": KNeighborsRegressor(),
    "SVR": SVR(),
}


def get_available_models(problem_type: str) -> list[str]:
    """Return names of available models for *problem_type*."""
    if problem_type == "classification":
        return list(_CLASSIFIERS.keys())
    return list(_REGRESSORS.keys())


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _get_model(problem_type: str, model_name: str) -> Any:
    registry = _CLASSIFIERS if problem_type == "classification" else _REGRESSORS
    if model_name not in registry:
        raise ValueError(f"Unknown model '{model_name}' for {problem_type}")
    import copy
    return copy.deepcopy(registry[model_name])


def _prepare_data(
    df: pd.DataFrame,
    target: str,
    feature_cols: list[str] | None,
    scale_features: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], LabelEncoder | None, StandardScaler | None]:
    """Prepare X, y arrays from DataFrame."""
    if feature_cols:
        cols = [c for c in feature_cols if c in df.columns and c != target]
    else:
        cols = [c for c in df.select_dtypes(include="number").columns if c != target]

    clean = df[cols + [target]].dropna()
    X = clean[cols].values.astype(float)
    y_raw = clean[target].values

    le: LabelEncoder | None = None
    if not pd.api.types.is_numeric_dtype(clean[target]):
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
    else:
        y = y_raw.astype(float)

    scaler: StandardScaler | None = None
    if scale_features:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X, y, cols, le, scaler


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _eval_classification(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    y_pred = model.predict(X_test)
    metrics: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "precision_weighted": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
    }
    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["confusion_matrix_labels"] = sorted(np.unique(np.concatenate([y_test, y_pred])).tolist())

    if hasattr(model, "predict_proba"):
        unique_classes = np.unique(y_test)
        if len(unique_classes) == 2:
            try:
                proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, proba)
                auc = round(float(roc_auc_score(y_test, proba)), 4)
                metrics["roc_auc"] = auc
                metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
            except Exception:
                pass

    return metrics


def _eval_regression(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    return {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(float(np.sqrt(mse)), 4),
        "r2": round(r2, 4),
        "mape": round(
            float(np.mean(np.abs((y_test - y_pred) / np.where(y_test == 0, 1, y_test))) * 100), 4
        ),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "residuals": (y_test - y_pred).tolist(),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_model(
    df: pd.DataFrame,
    target: str,
    model_name: str,
    problem_type: str = "regression",
    test_size: float = 0.2,
    feature_cols: list[str] | None = None,
    scale_features: bool = False,
    cv_folds: int = 5,
) -> dict[str, Any]:
    """Train a single model and return evaluation results.

    Parameters
    ----------
    df:
        Dataset.
    target:
        Target column.
    model_name:
        Name from :func:`get_available_models`.
    problem_type:
        ``"regression"`` or ``"classification"``.
    test_size:
        Fraction of data held out for testing.
    feature_cols:
        Explicit list of feature columns.  ``None`` uses all numeric columns.
    scale_features:
        Whether to apply :class:`~sklearn.preprocessing.StandardScaler`.
    cv_folds:
        Number of cross-validation folds (set to 0 to skip CV).

    Returns
    -------
    dict with keys ``model_name``, ``problem_type``, ``feature_cols``,
    ``train_size``, ``test_size``, ``metrics``, ``cv_scores``,
    ``feature_importance`` (if available).
    """
    X, y, used_cols, le, scaler = _prepare_data(df, target, feature_cols, scale_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42,
        stratify=y if problem_type == "classification" and len(np.unique(y)) <= 20 else None,
    )

    model = _get_model(problem_type, model_name)
    model.fit(X_train, y_train)

    if problem_type == "classification":
        metrics = _eval_classification(model, X_test, y_test)
    else:
        metrics = _eval_regression(model, X_test, y_test)

    cv_scores: dict[str, Any] = {}
    if cv_folds >= 2:
        scoring = "accuracy" if problem_type == "classification" else "r2"
        try:
            raw_cv = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring, n_jobs=-1)
            cv_scores = {
                "mean": round(float(raw_cv.mean()), 4),
                "std": round(float(raw_cv.std()), 4),
                "scores": [round(float(s), 4) for s in raw_cv],
                "scoring": scoring,
            }
        except Exception:
            pass

    feature_importance: dict[str, float] = {}
    if hasattr(model, "feature_importances_"):
        raw = model.feature_importances_
        feature_importance = dict(
            sorted(
                {col: round(float(v), 6) for col, v in zip(used_cols, raw)}.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )
    elif hasattr(model, "coef_"):
        coefs = np.abs(model.coef_).flatten()
        if len(coefs) == len(used_cols):
            feature_importance = dict(
                sorted(
                    {col: round(float(v), 6) for col, v in zip(used_cols, coefs)}.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )

    return {
        "model_name": model_name,
        "problem_type": problem_type,
        "target": target,
        "feature_cols": used_cols,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": metrics,
        "cv_scores": cv_scores,
        "feature_importance": feature_importance,
        "_model": model,
        "_scaler": scaler,
        "_le": le,
    }


def compare_models(
    df: pd.DataFrame,
    target: str,
    model_names: list[str],
    problem_type: str = "regression",
    test_size: float = 0.2,
    feature_cols: list[str] | None = None,
    scale_features: bool = False,
) -> list[dict[str, Any]]:
    """Train and evaluate multiple models, returning a ranked list.

    Returns
    -------
    list[dict]
        Each element is the result of :func:`train_model` (without the
        ``_model`` / ``_scaler`` / ``_le`` private keys), sorted by the
        primary metric descending.
    """
    results = []
    for name in model_names:
        try:
            r = train_model(
                df, target, name, problem_type, test_size, feature_cols, scale_features, cv_folds=5
            )
            # strip un-serialisable objects
            r.pop("_model", None)
            r.pop("_scaler", None)
            r.pop("_le", None)
            results.append(r)
        except Exception as exc:
            results.append({"model_name": name, "error": str(exc)})

    primary = "accuracy" if problem_type == "classification" else "r2"

    def _key(r: dict) -> float:
        m = r.get("metrics", {})
        return float(m.get(primary, -999))

    results.sort(key=_key, reverse=True)
    return results


def predict(
    result: dict[str, Any],
    df: pd.DataFrame,
) -> dict[str, Any]:
    """Apply a trained model result to new data.

    Parameters
    ----------
    result:
        Dict returned by :func:`train_model` (must contain ``_model``).
    df:
        DataFrame with the same feature columns as training.

    Returns
    -------
    dict with ``"predictions"`` list and optional ``"probabilities"``.
    """
    model = result["_model"]
    scaler = result.get("_scaler")
    le = result.get("_le")
    feature_cols = result["feature_cols"]

    X = df[feature_cols].fillna(0).values.astype(float)
    if scaler is not None:
        X = scaler.transform(X)

    preds = model.predict(X).tolist()
    out: dict[str, Any] = {"predictions": preds}

    if le is not None:
        out["predictions_labels"] = le.inverse_transform(np.array(preds, dtype=int)).tolist()

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X).tolist()
        out["probabilities"] = proba
        if hasattr(model, "classes_"):
            out["class_labels"] = (
                le.inverse_transform(model.classes_).tolist() if le else model.classes_.tolist()
            )

    return out


# ---------------------------------------------------------------------------
# Model serialisation
# ---------------------------------------------------------------------------


def export_model_bytes(result: dict[str, Any]) -> bytes:
    """Serialise the trained model (+ scaler/encoder) to bytes via joblib."""
    payload = {
        "model": result["_model"],
        "scaler": result.get("_scaler"),
        "le": result.get("_le"),
        "feature_cols": result["feature_cols"],
        "target": result["target"],
        "problem_type": result["problem_type"],
        "model_name": result["model_name"],
        "metrics": result.get("metrics", {}),
    }
    buf = io.BytesIO()
    joblib.dump(payload, buf)
    return buf.getvalue()


def export_model_b64(result: dict[str, Any]) -> str:
    """Return a base-64 encoded string of the serialised model payload."""
    return base64.b64encode(export_model_bytes(result)).decode()


def import_model_bytes(data: bytes) -> dict[str, Any]:
    """Load a model payload from raw bytes."""
    buf = io.BytesIO(data)
    return joblib.load(buf)


def import_model_b64(data_b64: str) -> dict[str, Any]:
    """Load a model payload from a base-64 encoded string."""
    return import_model_bytes(base64.b64decode(data_b64))


# ---------------------------------------------------------------------------
# Serialisable result helper
# ---------------------------------------------------------------------------


def serialisable_result(result: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *result* without un-serialisable sklearn objects.

    Stores the model as a base-64 string under the key ``"model_b64"``
    so the dict can be stored in a Dash dcc.Store.
    """
    clean = {k: v for k, v in result.items() if not k.startswith("_")}
    if "_model" in result:
        clean["model_b64"] = export_model_b64(result)
    return clean


def restore_result(clean: dict[str, Any]) -> dict[str, Any]:
    """Inverse of :func:`serialisable_result` — restore sklearn objects."""
    result = dict(clean)
    if "model_b64" in result:
        payload = import_model_b64(result.pop("model_b64"))
        result["_model"] = payload["model"]
        result["_scaler"] = payload.get("scaler")
        result["_le"] = payload.get("le")
    return result
