"""Preprocessing callbacks: imputation, outlier treatment, transformation, encoding."""

from __future__ import annotations

from typing import Any

import pandas as pd
import dash_mantine_components as dmc
from dash import Input, Output, State, callback, html, no_update
from dash.exceptions import PreventUpdate

from eda_core.preprocessing import (
    encode_categorical,
    get_preprocessing_summary,
    impute_missing,
    transform_column,
    treat_outliers,
)
from dashboard.components.cards import info_card


def _summary_badge(label: str, value: Any, color: str = "blue") -> dmc.Badge:
    return dmc.Badge(f"{label}: {value}", color=color, size="md", variant="light")


def register_preprocessing_callbacks(app: Any) -> None:
    """Register all preprocessing callbacks onto *app*."""

    # Populate preprocessing column selects when data is loaded
    @app.callback(
        Output("outlier-col-select", "data"),
        Output("transform-col-select", "data"),
        Output("encode-col-select", "data"),
        Input("data-store", "data"),
        prevent_initial_call=True,
    )
    def populate_preprocess_selects(json_data: str | None) -> tuple[list, list, list]:
        if not json_data:
            raise PreventUpdate
        df = pd.read_json(json_data, orient="split")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()
        return numeric_cols, numeric_cols, cat_cols

    # Preprocessing summary on data load
    @app.callback(
        Output("preprocessing-summary", "children"),
        Input("data-store", "data"),
        prevent_initial_call=True,
    )
    def update_preprocessing_summary(json_data: str | None) -> Any:
        if not json_data:
            raise PreventUpdate
        df = pd.read_json(json_data, orient="split")
        summary = get_preprocessing_summary(df)

        badges = [
            _summary_badge("Rows", summary["n_rows"]),
            _summary_badge("Cols", summary["n_cols"]),
            _summary_badge("Missing cols", len(summary["missing"]), "orange" if summary["missing"] else "green"),
            _summary_badge("Categorical cols", len(summary["categorical_cols"]), "violet"),
        ]

        skewed = {k: v for k, v in summary["skewness"].items() if abs(v) > 1}
        if skewed:
            skew_items = [
                html.Tr([html.Td(col), html.Td(f"{skew:.4f}")])
                for col, skew in skewed.items()
            ]
            skew_table = dmc.Table(
                [html.Thead(html.Tr([html.Th("Column"), html.Th("Skewness")])),
                 html.Tbody(skew_items)],
                withTableBorder=True, withColumnBorders=True,
            )
        else:
            skew_table = dmc.Text("No highly skewed columns (|skew| > 1).", c="dimmed", size="sm")

        return dmc.Card(
            [
                dmc.Text("Data Quality Summary", fw=600, size="lg", mb="sm"),
                dmc.Divider(mb="sm"),
                dmc.Group(badges, gap="xs", mb="md"),
                dmc.Text("Highly Skewed Columns", fw=500, mb="xs"),
                skew_table,
            ],
            withBorder=True,
            radius="md",
            p="md",
        )

    # Imputation
    @app.callback(
        Output("preprocessed-store", "data", allow_duplicate=True),
        Output("impute-result", "children"),
        Input("impute-apply-btn", "n_clicks"),
        State("impute-strategy-select", "value"),
        State("impute-constant-value", "value"),
        State("preprocessed-store", "data"),
        State("data-store", "data"),
        prevent_initial_call=True,
    )
    def apply_imputation(
        n_clicks: int | None,
        strategy: str,
        constant: str | None,
        proc_json: str | None,
        raw_json: str | None,
    ) -> tuple[Any, Any]:
        if not n_clicks:
            raise PreventUpdate
        source = proc_json or raw_json
        if not source:
            raise PreventUpdate
        df = pd.read_json(source, orient="split")
        fill_value: Any = 0
        if strategy == "constant" and constant:
            try:
                fill_value = float(constant)
            except ValueError:
                fill_value = constant
        df_new, summary = impute_missing(df, strategy=strategy, fill_value=fill_value)
        changed = summary.get("columns", {})
        rows_dropped = summary.get("rows_dropped", 0)

        if rows_dropped:
            msg = dmc.Alert(f"Dropped {rows_dropped:,} rows containing missing values.", color="orange")
        elif changed:
            items = [
                html.Li(f"'{col}': {info['missing_before']} → {info['missing_after']} missing")
                for col, info in changed.items()
            ]
            msg = dmc.Alert(
                html.Ul(items),
                title=f"Imputed {len(changed)} column(s) using '{strategy}'",
                color="green",
            )
        else:
            msg = dmc.Alert("No missing values to impute.", color="blue")

        return df_new.to_json(date_format="iso", orient="split"), msg

    # Outlier treatment
    @app.callback(
        Output("preprocessed-store", "data", allow_duplicate=True),
        Output("outlier-treatment-result", "children"),
        Input("outlier-apply-btn", "n_clicks"),
        State("outlier-col-select", "value"),
        State("outlier-method-select", "value"),
        State("outlier-action-select", "value"),
        State("preprocessed-store", "data"),
        State("data-store", "data"),
        prevent_initial_call=True,
    )
    def apply_outlier_treatment(
        n_clicks: int | None,
        column: str | None,
        method: str,
        action: str,
        proc_json: str | None,
        raw_json: str | None,
    ) -> tuple[Any, Any]:
        if not n_clicks or not column:
            raise PreventUpdate
        source = proc_json or raw_json
        if not source:
            raise PreventUpdate
        df = pd.read_json(source, orient="split")
        df_new, summary = treat_outliers(df, column, method=method, action=action)
        msg = dmc.Alert(
            f"Found {summary['outliers_found']} outlier(s) in '{column}' "
            f"[{summary['bounds']['lower']} – {summary['bounds']['upper']}]. "
            f"Action: {action}. Rows after: {summary['rows_after']:,}.",
            title="Outlier Treatment Applied",
            color="orange",
        )
        return df_new.to_json(date_format="iso", orient="split"), msg

    # Column transformation
    @app.callback(
        Output("preprocessed-store", "data", allow_duplicate=True),
        Output("transform-result", "children"),
        Input("transform-apply-btn", "n_clicks"),
        State("transform-col-select", "value"),
        State("transform-method-select", "value"),
        State("preprocessed-store", "data"),
        State("data-store", "data"),
        prevent_initial_call=True,
    )
    def apply_transformation(
        n_clicks: int | None,
        column: str | None,
        transformation: str,
        proc_json: str | None,
        raw_json: str | None,
    ) -> tuple[Any, Any]:
        if not n_clicks or not column:
            raise PreventUpdate
        source = proc_json or raw_json
        if not source:
            raise PreventUpdate
        df = pd.read_json(source, orient="split")
        try:
            df_new, summary = transform_column(df, column, transformation)
        except Exception as exc:
            return no_update, dmc.Alert(str(exc), title="Transformation Error", color="red")
        msg = dmc.Alert(
            f"Column '{column}' transformed via '{transformation}'. "
            f"Skewness: {summary['skew_before']} → {summary['skew_after']}. "
            f"New column: '{summary['output_column']}'.",
            title="Transformation Applied",
            color="teal",
        )
        return df_new.to_json(date_format="iso", orient="split"), msg

    # Categorical encoding
    @app.callback(
        Output("preprocessed-store", "data", allow_duplicate=True),
        Output("encode-result", "children"),
        Input("encode-apply-btn", "n_clicks"),
        State("encode-col-select", "value"),
        State("encode-method-select", "value"),
        State("preprocessed-store", "data"),
        State("data-store", "data"),
        prevent_initial_call=True,
    )
    def apply_encoding(
        n_clicks: int | None,
        column: str | None,
        method: str,
        proc_json: str | None,
        raw_json: str | None,
    ) -> tuple[Any, Any]:
        if not n_clicks or not column:
            raise PreventUpdate
        source = proc_json or raw_json
        if not source:
            raise PreventUpdate
        df = pd.read_json(source, orient="split")
        try:
            df_new, summary = encode_categorical(df, column, method)
        except Exception as exc:
            return no_update, dmc.Alert(str(exc), title="Encoding Error", color="red")
        msg = dmc.Alert(
            f"Column '{column}' encoded using '{method}'. "
            f"New columns: {', '.join(summary['new_columns'])}. "
            f"Unique values: {summary['unique_values']}.",
            title="Encoding Applied",
            color="violet",
        )
        return df_new.to_json(date_format="iso", orient="split"), msg
