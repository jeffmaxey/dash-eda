"""Visualisation callbacks for correlation, distribution, missing-value, bivariate, and multivariate tabs."""

from __future__ import annotations

from typing import Any

import pandas as pd
import dash_mantine_components as dmc
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from eda_core.analysis import get_bivariate_stats, get_multivariate_summary
from eda_core.charts import (
    bivariate_scatter,
    correlation_heatmap,
    distribution_plot,
    grouped_box_plot,
    missing_values_bar,
    outlier_box_plot,
)
from dashboard.components.cards import info_card, stat_card


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
        Input("distribution-show-outliers", "checked"),
        Input("data-store", "data"),
        prevent_initial_call=True,
    )
    def update_distribution(column: str | None, show_outliers: bool, json_data: str | None) -> Any:
        if not json_data or not column:
            raise PreventUpdate
        df = pd.read_json(json_data, orient="split")
        if column not in df.columns:
            raise PreventUpdate
        charts: list[Any] = [dcc.Graph(figure=distribution_plot(df, column), config={"displayModeBar": True})]
        if show_outliers and pd.api.types.is_numeric_dtype(df[column]):
            charts.append(dcc.Graph(figure=outlier_box_plot(df, column), config={"displayModeBar": True}))
        return dmc.Stack(charts, gap="md")

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

    # Bivariate tab: populate column selects
    @app.callback(
        Output("bivariate-x-select", "data"),
        Output("bivariate-y-select", "data"),
        Output("bivariate-color-select", "data"),
        Input("data-store", "data"),
        prevent_initial_call=True,
    )
    def populate_bivariate_selects(json_data: str | None) -> tuple[list, list, list]:
        if not json_data:
            raise PreventUpdate
        df = pd.read_json(json_data, orient="split")
        all_cols = df.columns.tolist()
        return all_cols, all_cols, all_cols

    # Bivariate chart
    @app.callback(
        Output("bivariate-content", "children"),
        Input("bivariate-x-select", "value"),
        Input("bivariate-y-select", "value"),
        Input("bivariate-color-select", "value"),
        Input("data-store", "data"),
        prevent_initial_call=True,
    )
    def update_bivariate(
        col_x: str | None,
        col_y: str | None,
        color_col: str | None,
        json_data: str | None,
    ) -> Any:
        if not json_data or not col_x or not col_y:
            raise PreventUpdate
        df = pd.read_json(json_data, orient="split")
        if col_x not in df.columns or col_y not in df.columns:
            raise PreventUpdate

        sections: list[Any] = []

        # Bivariate stats
        try:
            stats = get_bivariate_stats(df, col_x, col_y)
            btype = stats.get("type", "")
            badges: list[Any] = [dmc.Badge(f"N = {stats['n']:,}", color="blue")]
            if btype == "numeric_numeric":
                badges += [
                    dmc.Badge(f"Pearson r = {stats['pearson_r']:.4f}", color="teal"),
                    dmc.Badge(f"p = {stats['pearson_p']:.4f}", color="gray"),
                    dmc.Badge(f"Spearman ρ = {stats['spearman_r']:.4f}", color="violet"),
                ]
            elif btype == "numeric_categorical":
                badges += [
                    dmc.Badge(f"ANOVA F = {stats['anova_f']}", color="orange"),
                    dmc.Badge(f"p = {stats['anova_p']}", color="gray"),
                ]
            elif btype == "categorical_categorical":
                badges += [
                    dmc.Badge(f"χ² = {stats['chi2']:.4f}", color="orange"),
                    dmc.Badge(f"p = {stats['chi2_p']:.6f}", color="gray"),
                ]
            sections.append(dmc.Group(badges, mb="sm"))
        except Exception:
            pass

        # Chart
        x_numeric = pd.api.types.is_numeric_dtype(df[col_x])
        y_numeric = pd.api.types.is_numeric_dtype(df[col_y])
        try:
            if x_numeric and y_numeric:
                fig = bivariate_scatter(df, col_x, col_y, color_col)
            elif x_numeric != y_numeric:
                num_col = col_x if x_numeric else col_y
                cat_col = col_y if x_numeric else col_x
                fig = grouped_box_plot(df, num_col, cat_col)
            else:
                # Both categorical: show stacked bar via cross-tab
                ct = pd.crosstab(df[col_x], df[col_y])
                import plotly.express as px
                fig = px.bar(
                    ct.reset_index().melt(id_vars=col_x),
                    x=col_x,
                    y="value",
                    color="variable",
                    template="plotly_white",
                    title=f"{col_x} × {col_y} counts",
                ).to_dict()
                sections.append(dcc.Graph(figure=fig, config={"displayModeBar": True}))
                return dmc.Stack(sections, gap="md")

            sections.append(dcc.Graph(figure=fig, config={"displayModeBar": True}))
        except Exception as exc:
            sections.append(dmc.Alert(str(exc), color="red"))

        return dmc.Stack(sections, gap="md")

    # Multivariate tab
    @app.callback(
        Output("multivariate-content", "children"),
        Input("data-store", "data"),
        prevent_initial_call=True,
    )
    def update_multivariate(json_data: str | None) -> Any:
        if not json_data:
            raise PreventUpdate
        df = pd.read_json(json_data, orient="split")
        summary = get_multivariate_summary(df)

        sections: list[Any] = []

        # VIF table
        vif = summary.get("vif", {})
        if vif:
            vif_rows = [
                html.Tr(
                    [html.Td(col), html.Td(f"{v:.4f}"),
                     html.Td("⚠️ Multicollinear" if v > 5 else "✅")],
                )
                for col, v in sorted(vif.items(), key=lambda x: x[1], reverse=True)
            ]
            vif_table = dmc.Table(
                [
                    html.Thead(html.Tr([html.Th("Feature"), html.Th("VIF (approx)"), html.Th("Status")])),
                    html.Tbody(vif_rows),
                ],
                striped=True,
                highlightOnHover=True,
                withTableBorder=True,
                withColumnBorders=True,
            )
            sections.append(info_card("Variance Inflation Factor (VIF)", dmc.ScrollArea(vif_table, h=300)))

        # Eigenvalues / explained variance
        eigenvalues = summary.get("eigenvalues", [])
        explained = summary.get("explained_variance_pct", [])
        if eigenvalues:
            import plotly.graph_objects as go
            fig_eigen = go.Figure(
                [
                    go.Bar(
                        x=[f"PC{i+1}" for i in range(len(eigenvalues))],
                        y=explained,
                        marker_color="#4C6EF5",
                        name="Explained %",
                    ),
                    go.Scatter(
                        x=[f"PC{i+1}" for i in range(len(eigenvalues))],
                        y=[sum(explained[: i + 1]) for i in range(len(explained))],
                        mode="lines+markers",
                        name="Cumulative %",
                        yaxis="y2",
                        line={"color": "red"},
                    ),
                ]
            )
            fig_eigen.update_layout(
                title="Eigenvalue Scree Plot (Correlation Matrix)",
                template="plotly_white",
                yaxis={"title": "Explained Variance %"},
                yaxis2={"title": "Cumulative %", "overlaying": "y", "side": "right"},
                margin={"t": 60, "b": 60},
                height=380,
            )
            sections.append(dcc.Graph(figure=fig_eigen.to_dict(), config={"displayModeBar": False}))

        if not sections:
            return dmc.Text("Not enough numeric columns for multivariate analysis.", c="dimmed", ta="center")

        return dmc.Stack(sections, gap="md")
