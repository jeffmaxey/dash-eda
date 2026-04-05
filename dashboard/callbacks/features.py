"""Feature-analysis callbacks: selection, importance, engineering."""

from __future__ import annotations

from typing import Any

import pandas as pd
import dash_mantine_components as dmc
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from eda_core.features import (
    get_feature_target_stats,
    get_rf_feature_importance,
    select_k_best,
    select_by_variance,
)
from eda_core.charts import feature_importance_bar
from dashboard.components.cards import info_card


def register_features_callbacks(app: Any) -> None:
    """Register all feature-analysis callbacks onto *app*."""

    # Populate target selects for features tab
    @app.callback(
        Output("feature-target-select", "data"),
        Input("data-store", "data"),
        prevent_initial_call=True,
    )
    def populate_feature_target(json_data: str | None) -> list[str]:
        if not json_data:
            raise PreventUpdate
        df = pd.read_json(json_data, orient="split")
        return df.columns.tolist()

    # Main feature analysis
    @app.callback(
        Output("features-content", "children"),
        Input("feature-analyse-btn", "n_clicks"),
        State("feature-target-select", "value"),
        State("feature-problem-type", "value"),
        State("feature-k-select", "value"),
        State("preprocessed-store", "data"),
        State("data-store", "data"),
        prevent_initial_call=True,
    )
    def analyse_features(
        n_clicks: int | None,
        target: str | None,
        problem_type: str,
        k: int | None,
        proc_json: str | None,
        raw_json: str | None,
    ) -> Any:
        if not n_clicks or not target:
            raise PreventUpdate
        source = proc_json or raw_json
        if not source:
            raise PreventUpdate

        df = pd.read_json(source, orient="split")
        k_val = int(k) if k else 10
        sections: list[Any] = []

        # --- Feature-target correlations ---
        try:
            corr_stats = get_feature_target_stats(df, target, problem_type)
            pearson = corr_stats.get("pearson", {})
            mi = corr_stats.get("mutual_info", {})

            if pearson:
                sorted_pearson = dict(
                    sorted(pearson.items(), key=lambda x: abs(x[1]), reverse=True)
                )
                rows = [
                    html.Tr([
                        html.Td(col),
                        html.Td(f"{pearson.get(col, 0):.4f}"),
                        html.Td(f"{corr_stats['spearman'].get(col, 0):.4f}"),
                        html.Td(f"{mi.get(col, 0):.4f}"),
                    ])
                    for col in list(sorted_pearson.keys())[:20]
                ]
                corr_table = dmc.Table(
                    [
                        html.Thead(html.Tr([
                            html.Th("Feature"),
                            html.Th("Pearson r"),
                            html.Th("Spearman ρ"),
                            html.Th("Mutual Info"),
                        ])),
                        html.Tbody(rows),
                    ],
                    striped=True,
                    highlightOnHover=True,
                    withTableBorder=True,
                    withColumnBorders=True,
                )
                sections.append(
                    info_card("Feature–Target Correlations", dmc.ScrollArea(corr_table, h=320))
                )
        except Exception:
            pass

        # --- SelectKBest ---
        try:
            kbest = select_k_best(df, target, k=k_val, problem_type=problem_type)
            if kbest.get("selected_features"):
                scores = kbest["scores"]
                selected_set = set(kbest["selected_features"])
                rows = [
                    html.Tr(
                        [
                            html.Td(col),
                            html.Td(f"{scores.get(col, 0):.4f}"),
                            html.Td(f"{kbest['pvalues'].get(col, 1):.6f}"),
                            html.Td("✅" if col in selected_set else ""),
                        ],
                        style={"backgroundColor": "var(--mantine-color-blue-0)"} if col in selected_set else {},
                    )
                    for col in sorted(scores, key=lambda c: scores[c], reverse=True)
                ]
                kbest_table = dmc.Table(
                    [
                        html.Thead(html.Tr([
                            html.Th("Feature"),
                            html.Th("F-score"),
                            html.Th("p-value"),
                            html.Th("Selected"),
                        ])),
                        html.Tbody(rows),
                    ],
                    striped=True,
                    highlightOnHover=True,
                    withTableBorder=True,
                    withColumnBorders=True,
                )
                sections.append(
                    info_card(f"SelectKBest (top {k_val})", dmc.ScrollArea(kbest_table, h=320))
                )
        except Exception:
            pass

        # --- Variance threshold ---
        try:
            var_result = select_by_variance(df, threshold=0.0, exclude=[target])
            if var_result.get("variances"):
                variances = var_result["variances"]
                dropped_set = set(var_result.get("dropped", []))
                rows = [
                    html.Tr(
                        [html.Td(col), html.Td(f"{v:.6f}"),
                         html.Td("❌ Low variance" if col in dropped_set else "✅")],
                    )
                    for col, v in sorted(variances.items(), key=lambda x: x[1], reverse=True)
                ]
                var_table = dmc.Table(
                    [
                        html.Thead(html.Tr([html.Th("Feature"), html.Th("Variance"), html.Th("Status")])),
                        html.Tbody(rows),
                    ],
                    striped=True,
                    highlightOnHover=True,
                    withTableBorder=True,
                    withColumnBorders=True,
                )
                sections.append(info_card("Feature Variances", dmc.ScrollArea(var_table, h=300)))
        except Exception:
            pass

        # --- RF feature importance ---
        try:
            rf_result = get_rf_feature_importance(df, target, problem_type=problem_type, n_estimators=50)
            importances = rf_result.get("importances", {})
            if importances:
                top_imp = dict(list(importances.items())[:20])
                fig = feature_importance_bar(top_imp, "Random Forest Feature Importances")
                sections.append(
                    info_card(
                        "Random Forest Feature Importance",
                        dcc.Graph(figure=fig, config={"displayModeBar": False}),
                    )
                )
        except Exception:
            pass

        if not sections:
            return dmc.Text("Not enough numeric features for analysis. Ensure the dataset has numeric columns.", c="dimmed", ta="center")

        return dmc.Stack(sections, gap="md")
