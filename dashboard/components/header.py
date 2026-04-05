"""Header component for the Dash EDA AppShell."""

from __future__ import annotations

import dash_mantine_components as dmc
from dash import html


def create_header() -> dmc.AppShellHeader:
    """Return the AppShell header with title and colour-scheme toggle."""
    return dmc.AppShellHeader(
        dmc.Group(
            [
                dmc.Group(
                    [
                        html.Span("📊", style={"fontSize": "1.5rem"}),
                        dmc.Title("Dash EDA", order=3, style={"margin": 0}),
                    ],
                    gap="xs",
                ),
                dmc.ColorSchemeToggle(size="md"),
            ],
            justify="space-between",
            style={"height": "100%", "paddingLeft": "1rem", "paddingRight": "1rem"},
        ),
        style={"borderBottom": "1px solid var(--mantine-color-default-border)"},
    )
