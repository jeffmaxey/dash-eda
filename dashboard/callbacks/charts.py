"""Visualisation callbacks for correlation, distribution, and missing-value tabs."""

from __future__ import annotations

from typing import Any

import pandas as pd
import dash_mantine_components as dmc
from dash import Input, Output, dcc, html
from dash.exceptions import PreventUpdate

from eda_core.charts import (
    correlation_heatmap,
    distribution_plot,
    missing_values_bar,
)


def register_chart_callbacks(app: Any) -> None:
    """Register all chart callbacks onto *app*."""

    @app.callback(
        Output("correlation-content", "children"),
        Input("data-store", "data"),
        prevent_initial_call=True,
    )
    def update_correlation(json_data: str | None) -> Any:
        if not json_data:
            raise PreventUpdate
        df = pd.read_json(json_data, orient="split")
        fig_dict = correlation_heatmap(df)
        return dcc.Graph(figure=fig_dict, config={"displayModeBar": True})

    @app.callback(
        Output("distribution-content", "children"),
        Input("distribution-column-select", "value"),
        Input("data-store", "data"),
        prevent_initial_call=True,
    )
    def update_distribution(column: str | None, json_data: str | None) -> Any:
        if not json_data or not column:
            raise PreventUpdate
        df = pd.read_json(json_data, orient="split")
        if column not in df.columns:
            raise PreventUpdate
        fig_dict = distribution_plot(df, column)
        return dcc.Graph(figure=fig_dict, config={"displayModeBar": True})

    @app.callback(
        Output("missing-content", "children"),
        Input("data-store", "data"),
        prevent_initial_call=True,
    )
    def update_missing(json_data: str | None) -> Any:
        if not json_data:
            raise PreventUpdate
        df = pd.read_json(json_data, orient="split")
        bar_fig = missing_values_bar(df)

        total_missing = int(df.isnull().sum().sum())
        total_cells = df.shape[0] * df.shape[1]
        pct = round(total_missing / total_cells * 100, 2) if total_cells else 0.0

        summary = dmc.Group(
            [
                dmc.Badge(f"Total missing: {total_missing:,}", color="orange", size="lg"),
                dmc.Badge(f"Missing %: {pct}%", color="red" if pct > 5 else "green", size="lg"),
            ],
            mb="md",
        )
        return dmc.Stack(
            [summary, dcc.Graph(figure=bar_fig, config={"displayModeBar": True})],
            gap="md",
        )
