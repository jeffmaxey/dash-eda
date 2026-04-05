"""Modeling callbacks: experiment design, training, evaluation, comparison, predictions, export."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import dash_mantine_components as dmc
from dash import Input, Output, State, dcc, html, no_update
from dash.exceptions import PreventUpdate

from eda_core.modeling import (
    compare_models,
    export_model_bytes,
    get_available_models,
    restore_result,
    serialisable_result,
    train_model,
    predict,
)
from eda_core.charts import (
    actual_vs_predicted_plot,
    confusion_matrix_heatmap,
    feature_importance_bar,
    model_comparison_bar,
    residuals_plot,
    roc_curve_plot,
)
from dashboard.components.cards import info_card, stat_card


def register_modeling_callbacks(app: Any) -> None:
    """Register all modeling callbacks onto *app*."""

    # Populate target select and model list when data / problem type changes
    @app.callback(
        Output("model-target-select", "data"),
        Output("model-select", "data"),
        Input("data-store", "data"),
        Input("model-problem-type", "value"),
        prevent_initial_call=True,
    )
    def populate_model_selects(json_data: str | None, problem_type: str) -> tuple[list, list]:
        if not json_data:
            raise PreventUpdate
        df = pd.read_json(json_data, orient="split")
        cols = df.columns.tolist()
        models = get_available_models(problem_type or "regression")
        return cols, models

    # Train & evaluate
    @app.callback(
        Output("model-result-store", "data"),
        Output("modeling-content", "children"),
        Input("model-train-btn", "n_clicks"),
        State("model-target-select", "value"),
        State("model-problem-type", "value"),
        State("model-test-size", "value"),
        State("model-scale-features", "checked"),
        State("model-select", "value"),
        State("preprocessed-store", "data"),
        State("data-store", "data"),
        prevent_initial_call=True,
    )
    def train_and_evaluate(
        n_clicks: int | None,
        target: str | None,
        problem_type: str,
        test_size_str: str,
        scale: bool,
        selected_models: list[str] | None,
        proc_json: str | None,
        raw_json: str | None,
    ) -> tuple[Any, Any]:
        if not n_clicks or not target:
            raise PreventUpdate
        source = proc_json or raw_json
        if not source:
            raise PreventUpdate

        df = pd.read_json(source, orient="split")
        test_size = float(test_size_str or "0.2")
        models_to_train = selected_models or [get_available_models(problem_type)[0]]

        results: list[dict[str, Any]] = []
        errors: list[str] = []

        for model_name in models_to_train:
            try:
                r = train_model(
                    df,
                    target,
                    model_name,
                    problem_type,
                    test_size=test_size,
                    scale_features=scale,
                )
                results.append(r)
            except Exception as exc:
                errors.append(f"{model_name}: {exc}")

        if not results:
            msg = dmc.Alert("\n".join(errors) or "Training failed.", title="Error", color="red")
            return no_update, msg

        # Keep the best result (first, already sorted if multiple)
        best = results[0]
        serialised_best = serialisable_result(best)

        content_sections: list[Any] = []

        # Metric cards
        metrics = best.get("metrics", {})
        pt = best.get("problem_type", problem_type)
        if pt == "classification":
            metric_cards = [
                stat_card("Accuracy", f"{metrics.get('accuracy', 0):.4f}", color="blue"),
                stat_card("F1 (weighted)", f"{metrics.get('f1_weighted', 0):.4f}", color="teal"),
                stat_card("Precision", f"{metrics.get('precision_weighted', 0):.4f}", color="violet"),
                stat_card("Recall", f"{metrics.get('recall_weighted', 0):.4f}", color="indigo"),
            ]
            if "roc_auc" in metrics:
                metric_cards.append(stat_card("ROC-AUC", f"{metrics['roc_auc']:.4f}", color="green"))
        else:
            metric_cards = [
                stat_card("R²", f"{metrics.get('r2', 0):.4f}", color="blue"),
                stat_card("MAE", f"{metrics.get('mae', 0):.4f}", color="teal"),
                stat_card("RMSE", f"{metrics.get('rmse', 0):.4f}", color="orange"),
                stat_card("MAPE %", f"{metrics.get('mape', 0):.4f}", color="violet"),
            ]

        model_title = f"{best['model_name']} — {pt.title()}"
        content_sections.append(
            info_card(
                model_title,
                dmc.Stack(
                    [
                        dmc.Group(
                            [
                                dmc.Badge(f"Train: {best['train_samples']:,}", color="blue"),
                                dmc.Badge(f"Test: {best['test_samples']:,}", color="teal"),
                                dmc.Badge(f"Features: {len(best['feature_cols'])}", color="violet"),
                            ],
                            mb="sm",
                        ),
                        dmc.Group(metric_cards, grow=True, gap="sm"),
                    ],
                    gap="sm",
                ),
            )
        )

        # CV scores
        cv = best.get("cv_scores", {})
        if cv:
            content_sections.append(
                info_card(
                    f"Cross-Validation ({cv['scoring']})",
                    dmc.Group(
                        [
                            stat_card("CV Mean", f"{cv['mean']:.4f}", color="blue"),
                            stat_card("CV Std", f"{cv['std']:.4f}", color="orange"),
                        ]
                        + [stat_card(f"Fold {i+1}", f"{s:.4f}") for i, s in enumerate(cv["scores"])],
                        grow=True, gap="sm",
                    ),
                )
            )

        # Charts
        chart_cols: list[Any] = []

        if pt == "classification":
            cm = metrics.get("confusion_matrix")
            cm_labels = metrics.get("confusion_matrix_labels", [])
            if cm:
                chart_cols.append(
                    dmc.GridCol(
                        dcc.Graph(
                            figure=confusion_matrix_heatmap(cm, cm_labels),
                            config={"displayModeBar": False},
                        ),
                        span={"base": 12, "md": 6},
                    )
                )
            roc = metrics.get("roc_curve")
            auc = metrics.get("roc_auc")
            if roc and auc:
                chart_cols.append(
                    dmc.GridCol(
                        dcc.Graph(
                            figure=roc_curve_plot(roc["fpr"], roc["tpr"], auc),
                            config={"displayModeBar": False},
                        ),
                        span={"base": 12, "md": 6},
                    )
                )
        else:
            y_test = metrics.get("y_test", [])
            y_pred = metrics.get("y_pred", [])
            if y_test and y_pred:
                chart_cols.append(
                    dmc.GridCol(
                        dcc.Graph(
                            figure=actual_vs_predicted_plot(y_test, y_pred),
                            config={"displayModeBar": False},
                        ),
                        span={"base": 12, "md": 6},
                    )
                )
                chart_cols.append(
                    dmc.GridCol(
                        dcc.Graph(
                            figure=residuals_plot(y_test, y_pred),
                            config={"displayModeBar": False},
                        ),
                        span={"base": 12, "md": 6},
                    )
                )

        feat_imp = best.get("feature_importance", {})
        if feat_imp:
            chart_cols.append(
                dmc.GridCol(
                    dcc.Graph(
                        figure=feature_importance_bar(dict(list(feat_imp.items())[:20])),
                        config={"displayModeBar": False},
                    ),
                    span={"base": 12, "md": 6},
                )
            )

        if chart_cols:
            content_sections.append(
                dmc.Grid(chart_cols, gutter="md")
            )

        # If multiple models were trained, show comparison table
        if len(results) > 1:
            primary_metric = "accuracy" if pt == "classification" else "r2"
            comp_rows = [
                html.Tr([
                    html.Td(r["model_name"]),
                    html.Td(f"{r['metrics'].get(primary_metric, 0):.4f}"),
                    html.Td(f"{r['cv_scores'].get('mean', 0):.4f}" if r.get("cv_scores") else "—"),
                ])
                for r in results
            ]
            comp_table = dmc.Table(
                [
                    html.Thead(html.Tr([
                        html.Th("Model"),
                        html.Th(primary_metric.upper()),
                        html.Th("CV Mean"),
                    ])),
                    html.Tbody(comp_rows),
                ],
                striped=True,
                highlightOnHover=True,
                withTableBorder=True,
                withColumnBorders=True,
            )
            content_sections.append(info_card("Training Summary", comp_table))

        return json.dumps(serialised_best), dmc.Stack(content_sections, gap="md")

    # Compare all models
    @app.callback(
        Output("modeling-content", "children", allow_duplicate=True),
        Input("model-compare-btn", "n_clicks"),
        State("model-target-select", "value"),
        State("model-problem-type", "value"),
        State("model-test-size", "value"),
        State("model-scale-features", "checked"),
        State("preprocessed-store", "data"),
        State("data-store", "data"),
        prevent_initial_call=True,
    )
    def compare_all_models(
        n_clicks: int | None,
        target: str | None,
        problem_type: str,
        test_size_str: str,
        scale: bool,
        proc_json: str | None,
        raw_json: str | None,
    ) -> Any:
        if not n_clicks or not target:
            raise PreventUpdate
        source = proc_json or raw_json
        if not source:
            raise PreventUpdate

        df = pd.read_json(source, orient="split")
        test_size = float(test_size_str or "0.2")
        all_models = get_available_models(problem_type)

        try:
            results = compare_models(df, target, all_models, problem_type, test_size, scale_features=scale)
        except Exception as exc:
            return dmc.Alert(str(exc), title="Comparison Error", color="red")

        primary_metric = "accuracy" if problem_type == "classification" else "r2"

        comp_rows = []
        for r in results:
            if "error" in r:
                comp_rows.append(html.Tr([html.Td(r["model_name"]), html.Td("ERROR", colSpan=4)]))
                continue
            m = r.get("metrics", {})
            cv = r.get("cv_scores", {})
            comp_rows.append(
                html.Tr([
                    html.Td(r["model_name"]),
                    html.Td(f"{m.get(primary_metric, 0):.4f}"),
                    html.Td(f"{cv.get('mean', 0):.4f}" if cv else "—"),
                    html.Td(f"{cv.get('std', 0):.4f}" if cv else "—"),
                    html.Td(
                        str(r.get("train_samples", "?")) + " / " + str(r.get("test_samples", "?")),
                    ),
                ])
            )

        comp_table = dmc.Table(
            [
                html.Thead(html.Tr([
                    html.Th("Model"),
                    html.Th(primary_metric.upper()),
                    html.Th("CV Mean"),
                    html.Th("CV Std"),
                    html.Th("Train / Test"),
                ])),
                html.Tbody(comp_rows),
            ],
            striped=True,
            highlightOnHover=True,
            withTableBorder=True,
            withColumnBorders=True,
        )

        bar_fig = model_comparison_bar(results, primary_metric)

        return dmc.Stack(
            [
                info_card(f"Model Comparison — {problem_type.title()}", comp_table),
                dcc.Graph(figure=bar_fig, config={"displayModeBar": False}),
            ],
            gap="md",
        )

    # Predictions
    @app.callback(
        Output("predictions-content", "children"),
        Input("predict-btn", "n_clicks"),
        State("model-result-store", "data"),
        State("preprocessed-store", "data"),
        State("data-store", "data"),
        prevent_initial_call=True,
    )
    def generate_predictions(
        n_clicks: int | None,
        result_json: str | None,
        proc_json: str | None,
        raw_json: str | None,
    ) -> Any:
        if not n_clicks:
            raise PreventUpdate
        if not result_json:
            return dmc.Alert("No trained model found. Train a model on the Modeling tab first.", color="red")

        source = proc_json or raw_json
        if not source:
            raise PreventUpdate

        result = restore_result(json.loads(result_json))
        df = pd.read_json(source, orient="split")

        try:
            pred_out = predict(result, df)
        except Exception as exc:
            return dmc.Alert(str(exc), title="Prediction Error", color="red")

        preds = pred_out.get("predictions_labels") or pred_out["predictions"]
        target = result.get("target", "target")
        df_pred = df[result["feature_cols"]].copy()
        df_pred["prediction"] = preds

        import dash_ag_grid as dag
        col_defs = [{"field": col, "filter": True, "sortable": True} for col in df_pred.columns]
        grid = dag.AgGrid(
            columnDefs=col_defs,
            rowData=df_pred.head(500).to_dict("records"),
            defaultColDef={"resizable": True, "minWidth": 100},
            dashGridOptions={"pagination": True, "paginationPageSize": 50, "domLayout": "autoHeight"},
            style={"width": "100%"},
        )
        note = dmc.Text(f"Showing predictions for {min(500, len(df_pred)):,} rows.", size="xs", c="dimmed")
        return dmc.Stack([note, grid], gap="xs")

    # Model download
    @app.callback(
        Output("model-download", "data"),
        Input("model-download-btn", "n_clicks"),
        State("model-result-store", "data"),
        prevent_initial_call=True,
    )
    def download_model(n_clicks: int | None, result_json: str | None) -> Any:
        if not n_clicks or not result_json:
            raise PreventUpdate
        result = restore_result(json.loads(result_json))
        model_bytes = export_model_bytes(result)
        model_name = result.get("model_name", "model").replace(" ", "_").lower()
        return dcc.send_bytes(model_bytes, filename=f"{model_name}.joblib")
