"""Plotly chart-builder functions.

All public functions accept a pandas DataFrame (and optional column names)
and return a Plotly figure serialised as a plain dict (fig.to_dict()).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from eda_core.analysis import get_correlation_matrix, get_column_distribution


def _empty_figure(message: str = "No data available") -> dict[str, Any]:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 16, "color": "#666"},
    )
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        template="plotly_white",
        margin={"t": 40, "b": 20},
    )
    return fig.to_dict()


def correlation_heatmap(df: pd.DataFrame) -> dict[str, Any]:
    """Heatmap of the Pearson correlation matrix for numeric columns."""
    data = get_correlation_matrix(df)
    cols = data["columns"]
    if not cols or (len(cols) == 1 and not data["values"]):
        return _empty_figure("Not enough numeric columns for correlation")

    values = data["values"]
    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=cols,
            y=cols,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in values],
            texttemplate="%{text}",
            hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>r = %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Correlation Matrix",
        template="plotly_white",
        margin={"t": 60, "b": 40, "l": 40, "r": 40},
        height=max(400, len(cols) * 40),
    )
    return fig.to_dict()


def missing_values_bar(df: pd.DataFrame) -> dict[str, Any]:
    """Horizontal bar chart of missing-value percentage per column."""
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=True)

    if missing_pct.empty:
        return _empty_figure("No missing values 🎉")

    fig = go.Figure(
        go.Bar(
            x=missing_pct.values.tolist(),
            y=missing_pct.index.tolist(),
            orientation="h",
            marker_color="#4C6EF5",
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="Missing Values by Column (%)",
        xaxis_title="Missing %",
        template="plotly_white",
        margin={"t": 60, "b": 40, "l": 120, "r": 40},
        height=max(300, len(missing_pct) * 30 + 100),
    )
    return fig.to_dict()


def distribution_plot(df: pd.DataFrame, column: str) -> dict[str, Any]:
    """Histogram for numeric columns, bar chart for categorical."""
    dist = get_column_distribution(df, column)

    if dist["kind"] == "numeric":
        bins = dist["bins"]
        counts = dist["counts"]
        bin_mids = [(bins[i] + bins[i + 1]) / 2 for i in range(len(counts))]
        fig = go.Figure(
            go.Bar(
                x=bin_mids,
                y=counts,
                marker_color="#4C6EF5",
                hovertemplate="Value: %{x:.2f}<br>Count: %{y}<extra></extra>",
            )
        )
        stats = dist["stats"]
        fig.add_vline(
            x=stats["mean"],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {stats['mean']}",
        )
        fig.update_layout(
            title=f"Distribution of '{column}'",
            xaxis_title=column,
            yaxis_title="Count",
            template="plotly_white",
            margin={"t": 60, "b": 40},
        )
    else:
        vc = dist["value_counts"]
        labels = list(vc.keys())[:20]
        values_list = [vc[k] for k in labels]
        fig = go.Figure(
            go.Bar(
                x=labels,
                y=values_list,
                marker_color="#4C6EF5",
                hovertemplate="%{x}: %{y}<extra></extra>",
            )
        )
        fig.update_layout(
            title=f"Value Counts for '{column}'",
            xaxis_title=column,
            yaxis_title="Count",
            template="plotly_white",
            margin={"t": 60, "b": 80},
        )

    return fig.to_dict()


def scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
) -> dict[str, Any]:
    """Scatter plot of two numeric columns with optional colour grouping."""
    kwargs: dict[str, Any] = {"x": x_col, "y": y_col, "template": "plotly_white"}
    if color_col and color_col in df.columns:
        kwargs["color"] = color_col
    fig = px.scatter(df, **kwargs)
    fig.update_layout(
        title=f"{x_col} vs {y_col}",
        margin={"t": 60, "b": 40},
    )
    return fig.to_dict()


def box_plot(
    df: pd.DataFrame,
    column: str,
    group_col: str | None = None,
) -> dict[str, Any]:
    """Box plot for a numeric column with optional categorical grouping."""
    kwargs: dict[str, Any] = {"y": column, "template": "plotly_white"}
    if group_col and group_col in df.columns:
        kwargs["x"] = group_col
    fig = px.box(df, **kwargs)
    fig.update_layout(
        title=f"Box Plot – {column}",
        margin={"t": 60, "b": 40},
    )
    return fig.to_dict()


def time_series_plot(df: pd.DataFrame, x_col: str, y_col: str) -> dict[str, Any]:
    """Line chart for time-series data."""
    fig = px.line(
        df.sort_values(x_col),
        x=x_col,
        y=y_col,
        template="plotly_white",
    )
    fig.update_layout(
        title=f"{y_col} over {x_col}",
        margin={"t": 60, "b": 40},
    )
    return fig.to_dict()


# ---------------------------------------------------------------------------
# Outlier visualisation
# ---------------------------------------------------------------------------


def outlier_box_plot(df: pd.DataFrame, column: str) -> dict[str, Any]:
    """Box plot highlighting IQR outliers for *column*."""
    from eda_core.analysis import detect_outliers

    info = detect_outliers(df, column)
    series = df[column].dropna()
    lower = info["bounds"]["lower"]
    upper = info["bounds"]["upper"]
    is_outlier = (series < lower) | (series > upper)

    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=series.values.tolist(),
            name=column,
            marker_color="#4C6EF5",
            boxpoints="outliers",
            hovertemplate="%{y:.4f}<extra></extra>",
        )
    )
    fig.add_hline(y=lower, line_dash="dash", line_color="red", annotation_text=f"Lower: {lower}")
    fig.add_hline(y=upper, line_dash="dash", line_color="red", annotation_text=f"Upper: {upper}")
    fig.update_layout(
        title=f"Box Plot – {column} ({int(is_outlier.sum())} outliers)",
        template="plotly_white",
        margin={"t": 60, "b": 40},
        height=400,
    )
    return fig.to_dict()


# ---------------------------------------------------------------------------
# Bivariate charts
# ---------------------------------------------------------------------------


def bivariate_scatter(
    df: pd.DataFrame,
    col_x: str,
    col_y: str,
    color_col: str | None = None,
    trendline: bool = True,
) -> dict[str, Any]:
    """Scatter plot with optional OLS trendline for two numeric columns."""
    kwargs: dict[str, Any] = {
        "x": col_x,
        "y": col_y,
        "template": "plotly_white",
        "opacity": 0.7,
    }
    if color_col and color_col in df.columns:
        kwargs["color"] = color_col
    if trendline:
        kwargs["trendline"] = "ols"
    try:
        fig = px.scatter(df.dropna(subset=[col_x, col_y]), **kwargs)
    except Exception:
        kwargs.pop("trendline", None)
        fig = px.scatter(df.dropna(subset=[col_x, col_y]), **kwargs)
    fig.update_layout(
        title=f"{col_x} vs {col_y}",
        margin={"t": 60, "b": 40},
        height=450,
    )
    return fig.to_dict()


def grouped_box_plot(
    df: pd.DataFrame,
    num_col: str,
    cat_col: str,
) -> dict[str, Any]:
    """Box plot of *num_col* grouped by *cat_col*."""
    fig = px.box(
        df.dropna(subset=[num_col, cat_col]),
        x=cat_col,
        y=num_col,
        template="plotly_white",
        color=cat_col,
    )
    fig.update_layout(
        title=f"{num_col} by {cat_col}",
        margin={"t": 60, "b": 60},
        height=420,
        showlegend=False,
    )
    return fig.to_dict()


# ---------------------------------------------------------------------------
# Feature importance chart
# ---------------------------------------------------------------------------


def feature_importance_bar(
    importances: dict[str, float],
    title: str = "Feature Importances",
) -> dict[str, Any]:
    """Horizontal bar chart of feature importances."""
    if not importances:
        return _empty_figure("No feature importances available")

    items = sorted(importances.items(), key=lambda x: x[1])
    features = [i[0] for i in items]
    values = [i[1] for i in items]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=features,
            orientation="h",
            marker_color="#4C6EF5",
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        template="plotly_white",
        margin={"t": 60, "b": 40, "l": 140, "r": 40},
        height=max(300, len(features) * 25 + 100),
    )
    return fig.to_dict()


# ---------------------------------------------------------------------------
# Model evaluation charts
# ---------------------------------------------------------------------------


def confusion_matrix_heatmap(cm: list[list[int]], labels: list[Any]) -> dict[str, Any]:
    """Heatmap of a confusion matrix."""
    str_labels = [str(lbl) for lbl in labels]
    text = [[str(v) for v in row] for row in cm]
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=str_labels,
            y=str_labels,
            colorscale="Blues",
            text=text,
            texttemplate="%{text}",
            hovertemplate="Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template="plotly_white",
        margin={"t": 60, "b": 60, "l": 60, "r": 40},
        height=max(300, len(labels) * 60 + 100),
    )
    return fig.to_dict()


def roc_curve_plot(fpr: list[float], tpr: list[float], auc: float) -> dict[str, Any]:
    """ROC curve with AUC annotation."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC (AUC={auc:.3f})",
            line={"color": "#4C6EF5", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line={"color": "gray", "dash": "dash"},
        )
    )
    fig.update_layout(
        title=f"ROC Curve  (AUC = {auc:.4f})",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        margin={"t": 60, "b": 60},
        height=400,
    )
    return fig.to_dict()


def residuals_plot(y_test: list[float], y_pred: list[float]) -> dict[str, Any]:
    """Scatter of residuals vs predicted values."""
    y_test_arr = list(y_test)
    y_pred_arr = list(y_pred)
    residuals = [a - b for a, b in zip(y_test_arr, y_pred_arr)]
    fig = go.Figure(
        go.Scatter(
            x=y_pred_arr,
            y=residuals,
            mode="markers",
            marker={"color": "#4C6EF5", "opacity": 0.6, "size": 6},
            hovertemplate="Predicted: %{x:.3f}<br>Residual: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(
        title="Residuals vs Predicted",
        xaxis_title="Predicted",
        yaxis_title="Residual",
        template="plotly_white",
        margin={"t": 60, "b": 60},
        height=380,
    )
    return fig.to_dict()


def actual_vs_predicted_plot(y_test: list[float], y_pred: list[float]) -> dict[str, Any]:
    """Actual vs Predicted scatter with identity line."""
    y_test_arr = list(y_test)
    y_pred_arr = list(y_pred)
    mn = min(min(y_test_arr), min(y_pred_arr))
    mx = max(max(y_test_arr), max(y_pred_arr))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_pred_arr,
            y=y_test_arr,
            mode="markers",
            marker={"color": "#4C6EF5", "opacity": 0.6, "size": 6},
            name="Samples",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[mn, mx],
            y=[mn, mx],
            mode="lines",
            name="Perfect fit",
            line={"color": "red", "dash": "dash"},
        )
    )
    fig.update_layout(
        title="Actual vs Predicted",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template="plotly_white",
        margin={"t": 60, "b": 60},
        height=400,
    )
    return fig.to_dict()


def model_comparison_bar(results: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Bar chart comparing multiple models on a single metric."""
    names = []
    values = []
    for r in results:
        if "error" in r:
            continue
        names.append(r.get("model_name", "?"))
        values.append(r.get("metrics", {}).get(metric, 0))

    if not names:
        return _empty_figure("No model results available")

    max_val = max(values) if values else 0
    colors = ["#4C6EF5" if v == max_val else "#ADB5BD" for v in values]
    fig = go.Figure(
        go.Bar(
            x=names,
            y=values,
            marker_color=colors,
            hovertemplate="%{x}: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Model Comparison – {metric}",
        xaxis_title="Model",
        yaxis_title=metric,
        template="plotly_white",
        margin={"t": 60, "b": 80},
        height=380,
    )
    return fig.to_dict()
