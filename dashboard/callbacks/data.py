"""Data-management callbacks: upload, store, overview, stats, data grid."""

from __future__ import annotations

from typing import Any

import pandas as pd
import dash_ag_grid as dag
import dash_mantine_components as dmc
from dash import Input, Output, State, callback, html, no_update
from dash.exceptions import PreventUpdate

from eda_core.analysis import parse_upload
from eda_core.analysis import get_overview, get_summary_stats
from dashboard.components.cards import create_overview_cards, info_card


def _fmt_cell(value: Any) -> str:
    """Format a table cell value as a short string."""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


_MAX_GRID_ROWS = 10_000


def register_data_callbacks(app: Any) -> None:
    """Register all data-management callbacks onto *app*."""

    @app.callback(
        Output("data-store", "data"),
        Output("upload-status", "children"),
        Input("upload-data", "contents"),
        State("upload-data", "filename"),
        prevent_initial_call=True,
    )
    def store_upload(contents: str | None, filename: str | None) -> tuple[Any, Any]:
        if contents is None:
            raise PreventUpdate

        try:
            df = parse_upload(contents, filename or "upload.csv")
        except Exception as exc:
            alert = dmc.Alert(
                f"Failed to parse '{filename}': {exc}",
                title="Upload Error",
                color="red",
                withCloseButton=True,
            )
            return no_update, alert

        return df.to_json(date_format="iso", orient="split"), dmc.Alert(
            f"✅ Loaded '{filename}' — {df.shape[0]:,} rows × {df.shape[1]} columns",
            color="green",
            withCloseButton=True,
        )

    @app.callback(
        Output("overview-content", "children"),
        Output("distribution-column-select", "data"),
        Input("data-store", "data"),
        prevent_initial_call=True,
    )
    def update_overview(json_data: str | None) -> tuple[Any, list[str]]:
        if not json_data:
            raise PreventUpdate

        df = pd.read_json(json_data, orient="split")
        overview = get_overview(df)

        cards_row = dmc.Group(
            create_overview_cards(overview),
            grow=True,
            gap="sm",
            mb="md",
        )

        # Column dtype table
        dtype_rows = [
            html.Tr(
                [
                    html.Td(col),
                    html.Td(str(dtype)),
                    html.Td(str(overview["missing_counts"].get(col, 0))),
                    html.Td(f"{overview['missing_pct'].get(col, 0.0):.1f}%"),
                ]
            )
            for col, dtype in overview["dtypes"].items()
        ]

        dtype_table = dmc.Table(
            [
                html.Thead(
                    html.Tr([
                        html.Th("Column"),
                        html.Th("dtype"),
                        html.Th("Missing"),
                        html.Th("Missing %"),
                    ])
                ),
                html.Tbody(dtype_rows),
            ],
            striped=True,
            highlightOnHover=True,
            withTableBorder=True,
            withColumnBorders=True,
        )

        content = dmc.Stack(
            [
                cards_row,
                info_card("Column Details", dmc.ScrollArea(dtype_table, h=300)),
            ],
            gap="md",
        )

        col_options = df.columns.tolist()
        return content, col_options

    @app.callback(
        Output("stats-content", "children"),
        Input("data-store", "data"),
        prevent_initial_call=True,
    )
    def update_stats(json_data: str | None) -> Any:
        if not json_data:
            raise PreventUpdate

        df = pd.read_json(json_data, orient="split")
        stats = get_summary_stats(df)

        sections: list[Any] = []

        # Numeric summary
        num_summary = stats.get("numeric_summary", {})
        if num_summary:
            # Transpose so rows = stats (count, mean …) and columns = column names
            summary_df = pd.DataFrame(num_summary)
            header_cells = [html.Th("Stat")] + [html.Th(c) for c in summary_df.columns]
            body_rows = [
                html.Tr([html.Td(idx)] + [html.Td(_fmt_cell(v)) for v in row])
                for idx, row in summary_df.iterrows()
            ]
            num_table = dmc.Table(
                [html.Thead(html.Tr(header_cells)), html.Tbody(body_rows)],
                striped=True,
                highlightOnHover=True,
                withTableBorder=True,
                withColumnBorders=True,
            )
            sections.append(info_card("Numeric Summary", dmc.ScrollArea(num_table, h=320)))

        # Categorical summary
        cat_summary = stats.get("categorical_summary", {})
        if cat_summary:
            cat_blocks = []
            for col, vc in cat_summary.items():
                rows = [html.Tr([html.Td(str(k)), html.Td(str(v))]) for k, v in vc.items()]
                tbl = dmc.Table(
                    [
                        html.Thead(html.Tr([html.Th("Value"), html.Th("Count")])),
                        html.Tbody(rows),
                    ],
                    withTableBorder=True,
                    withColumnBorders=True,
                )
                cat_blocks.append(info_card(f"'{col}' — top values", tbl))

            sections.append(
                dmc.Grid(
                    [dmc.GridCol(b, span={"base": 12, "sm": 6, "lg": 4}) for b in cat_blocks],
                    gutter="md",
                )
            )

        return dmc.Stack(sections, gap="md") if sections else dmc.Text("No columns to summarise.", c="dimmed")

    @app.callback(
        Output("datagrid-content", "children"),
        Input("data-store", "data"),
        prevent_initial_call=True,
    )
    def update_datagrid(json_data: str | None) -> Any:
        if not json_data:
            raise PreventUpdate

        df = pd.read_json(json_data, orient="split")
        # Limit preview rows for browser performance
        preview = df.head(_MAX_GRID_ROWS)
        col_defs = [{"field": col, "filter": True, "sortable": True} for col in preview.columns]
        row_data = preview.to_dict("records")

        grid = dag.AgGrid(
            id="main-ag-grid",
            columnDefs=col_defs,
            rowData=row_data,
            defaultColDef={
                "resizable": True,
                "minWidth": 100,
            },
            dashGridOptions={
                "pagination": True,
                "paginationPageSize": 50,
                "domLayout": "autoHeight",
            },
            style={"width": "100%"},
        )

        note = dmc.Text(
            f"Showing {len(preview):,} of {len(df):,} rows",
            size="xs",
            c="dimmed",
            mb="xs",
        )
        return dmc.Stack([note, grid], gap="xs")
