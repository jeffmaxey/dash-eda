"""Main layout for the Dash EDA application."""

from __future__ import annotations

import dash_mantine_components as dmc
from dash import dcc, html

from dashboard.components.header import create_header

# Navigation items: (label, tab_value, icon)
NAV_ITEMS = [
    ("Overview", "overview", "🏠"),
    ("Statistics", "statistics", "📈"),
    ("Correlations", "correlations", "🔗"),
    ("Distributions", "distributions", "📊"),
    ("Missing Values", "missing", "❓"),
    ("Data Grid", "datagrid", "🗂️"),
]


def _nav_item(label: str, value: str, icon: str) -> html.Div:
    return html.Div(
        dmc.NavLink(
            label=dmc.Group(
                [html.Span(icon, style={"fontSize": "1rem"}), dmc.Text(label, size="sm")],
                gap="xs",
            ),
            id=f"nav-{value}",
            href=f"#{value}",
            style={"borderRadius": "var(--mantine-radius-sm)"},
        )
    )


def create_layout() -> dmc.MantineProvider:
    """Build and return the complete application layout."""
    navbar_content = dmc.AppShellNavbar(
        dmc.Stack(
            [
                dmc.AppShellSection(
                    dmc.Text("Navigation", size="xs", c="dimmed", fw=600, tt="uppercase", p="sm"),
                ),
                dmc.AppShellSection(
                    dmc.Stack(
                        [_nav_item(label, val, icon) for label, val, icon in NAV_ITEMS],
                        gap=2,
                        p="xs",
                    ),
                    grow=True,
                ),
                dmc.AppShellSection(
                    dmc.Text(
                        "Dash EDA v0.1.0",
                        size="xs",
                        c="dimmed",
                        ta="center",
                        p="sm",
                    ),
                ),
            ],
            h="100%",
            gap=0,
        ),
        style={"borderRight": "1px solid var(--mantine-color-default-border)"},
    )

    upload_section = dmc.Paper(
        dcc.Upload(
            id="upload-data",
            children=dmc.Stack(
                [
                    html.Span("📂", style={"fontSize": "2.5rem"}),
                    dmc.Text("Drag & drop a CSV or Excel file here", size="md", fw=500),
                    dmc.Text("or click to browse", size="sm", c="dimmed"),
                    dmc.Badge("CSV • XLS • XLSX", color="blue", variant="light"),
                ],
                align="center",
                gap="xs",
            ),
            style={
                "width": "100%",
                "border": "2px dashed var(--mantine-color-blue-4)",
                "borderRadius": "var(--mantine-radius-md)",
                "padding": "2rem",
                "cursor": "pointer",
                "textAlign": "center",
                "backgroundColor": "var(--mantine-color-blue-0)",
            },
            multiple=False,
            accept=".csv,.xls,.xlsx,.xlsm,.xlsb",
        ),
        radius="md",
        mb="md",
    )

    tabs = dmc.Tabs(
        [
            dmc.TabsList(
                [
                    dmc.TabsTab(label, value=val)
                    for label, val, _ in NAV_ITEMS
                ],
                grow=True,
            ),
            # Overview
            dmc.TabsPanel(
                dmc.LoadingOverlay(
                    html.Div(
                        dmc.Stack(
                            [
                                dmc.Text(
                                    "Upload a dataset to begin exploration.",
                                    c="dimmed",
                                    ta="center",
                                    mt="xl",
                                )
                            ],
                            align="center",
                        ),
                        id="overview-content",
                    ),
                    visible=False,
                    id="overview-loading",
                    overlayProps={"radius": "md"},
                ),
                value="overview",
                pt="md",
            ),
            # Statistics
            dmc.TabsPanel(
                html.Div(
                    dmc.Text("Upload a dataset to view statistics.", c="dimmed", ta="center", mt="xl"),
                    id="stats-content",
                ),
                value="statistics",
                pt="md",
            ),
            # Correlations
            dmc.TabsPanel(
                html.Div(
                    dmc.Text("Upload a dataset to view correlations.", c="dimmed", ta="center", mt="xl"),
                    id="correlation-content",
                ),
                value="correlations",
                pt="md",
            ),
            # Distributions
            dmc.TabsPanel(
                dmc.Stack(
                    [
                        dmc.Select(
                            id="distribution-column-select",
                            label="Select column",
                            placeholder="Choose a column…",
                            data=[],
                            w=300,
                        ),
                        html.Div(
                            dmc.Text("Upload a dataset to view distributions.", c="dimmed", ta="center", mt="xl"),
                            id="distribution-content",
                        ),
                    ],
                    gap="md",
                ),
                value="distributions",
                pt="md",
            ),
            # Missing Values
            dmc.TabsPanel(
                html.Div(
                    dmc.Text("Upload a dataset to analyse missing values.", c="dimmed", ta="center", mt="xl"),
                    id="missing-content",
                ),
                value="missing",
                pt="md",
            ),
            # Data Grid
            dmc.TabsPanel(
                html.Div(
                    dmc.Text("Upload a dataset to view the data grid.", c="dimmed", ta="center", mt="xl"),
                    id="datagrid-content",
                ),
                value="datagrid",
                pt="md",
            ),
        ],
        value="overview",
        id="main-tabs",
    )

    main_content = dmc.AppShellMain(
        dmc.ScrollArea(
            dmc.Container(
                dmc.Stack(
                    [
                        # Status / alert area
                        html.Div(id="upload-status"),
                        upload_section,
                        tabs,
                    ],
                    gap="md",
                ),
                fluid=True,
                p="md",
            ),
            style={"height": "100%"},
        )
    )

    app_shell = dmc.AppShell(
        [
            create_header(),
            navbar_content,
            main_content,
        ],
        navbar={"width": 250, "breakpoint": "sm", "collapsed": {"mobile": True}},
        header={"height": 60},
        padding="0",
    )

    return dmc.MantineProvider(
        children=[
            dcc.Store(id="data-store", storage_type="memory"),
            app_shell,
        ],
        theme={
            "primaryColor": "blue",
            "fontFamily": "'Inter', sans-serif",
        },
        defaultColorScheme="light",
        id="mantine-provider",
    )
