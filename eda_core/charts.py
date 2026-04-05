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
