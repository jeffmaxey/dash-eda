"""Reusable card components for the Dash EDA dashboard."""

from __future__ import annotations

from typing import Any

import dash_mantine_components as dmc
from dash import html


def stat_card(
    title: str,
    value: Any,
    description: str | None = None,
    color: str = "blue",
) -> dmc.Card:
    """A compact statistics card displaying a single metric."""
    children = [
        dmc.Text(title, size="sm", c="dimmed", tt="uppercase", fw=600),
        dmc.Text(str(value), size="xl", fw=700, c=color),
    ]
    if description:
        children.append(dmc.Text(description, size="xs", c="dimmed", mt=4))
    return dmc.Card(
        children=dmc.Stack(children, gap=4),
        withBorder=True,
        radius="md",
        p="md",
        style={"flex": "1 1 160px"},
    )


def info_card(title: str, children: Any) -> dmc.Card:
    """A titled wrapper card."""
    return dmc.Card(
        children=[
            dmc.Text(title, fw=600, size="lg", mb="sm"),
            dmc.Divider(mb="sm"),
            children,
        ],
        withBorder=True,
        radius="md",
        p="md",
    )


def create_overview_cards(overview: dict[str, Any]) -> list[dmc.Card]:
    """Build a row of stat cards from an overview dict."""
    shape = overview.get("shape", {})
    total_missing = sum(overview.get("missing_counts", {}).values())
    total_cells = shape.get("rows", 0) * shape.get("columns", 1)
    missing_pct = round(total_missing / total_cells * 100, 2) if total_cells else 0.0

    return [
        stat_card("Rows", f"{shape.get('rows', 0):,}", "total records", "blue"),
        stat_card("Columns", shape.get("columns", 0), "features", "violet"),
        stat_card(
            "Missing",
            f"{total_missing:,}",
            f"{missing_pct}% of all cells",
            "orange" if total_missing else "green",
        ),
        stat_card(
            "Duplicates",
            f"{overview.get('duplicate_rows', 0):,}",
            "duplicate rows",
            "red" if overview.get("duplicate_rows", 0) else "green",
        ),
        stat_card(
            "Memory",
            f"{overview.get('memory_usage_mb', 0)} MB",
            "in-memory size",
            "teal",
        ),
    ]
