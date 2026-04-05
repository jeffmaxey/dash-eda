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
    ("Bivariate", "bivariate", "🔬"),
    ("Multivariate", "multivariate", "🧮"),
    ("Missing Values", "missing", "❓"),
    ("Preprocessing", "preprocessing", "⚙️"),
    ("Features", "features", "🧩"),
    ("Modeling", "modeling", "🤖"),
    ("Predictions", "predictions", "🎯"),
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
                    dmc.ScrollArea(
                        dmc.Stack(
                            [_nav_item(label, val, icon) for label, val, icon in NAV_ITEMS],
                            gap=2,
                            p="xs",
                        ),
                        style={"height": "calc(100vh - 140px)"},
                    ),
                    grow=True,
                ),
                dmc.AppShellSection(
                    dmc.Text(
                        "Dash EDA v0.2.0",
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
                grow=False,
            ),
            # Overview
            dmc.TabsPanel(
                dmc.Box(
                    [
                        dmc.LoadingOverlay(
                            visible=False,
                            id="overview-loading",
                            overlayProps={"radius": "md"},
                            zIndex=10,
                        ),
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
                    ],
                    pos="relative",
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
                        dmc.Group(
                            [
                                dmc.Select(
                                    id="distribution-column-select",
                                    label="Select column",
                                    placeholder="Choose a column…",
                                    data=[],
                                    w=300,
                                ),
                                dmc.Switch(
                                    id="distribution-show-outliers",
                                    label="Show outlier box plot",
                                    checked=False,
                                    mt="lg",
                                ),
                            ],
                            align="flex-end",
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
            # Bivariate
            dmc.TabsPanel(
                dmc.Stack(
                    [
                        dmc.Group(
                            [
                                dmc.Select(
                                    id="bivariate-x-select",
                                    label="X column",
                                    placeholder="Choose X…",
                                    data=[],
                                    w=240,
                                ),
                                dmc.Select(
                                    id="bivariate-y-select",
                                    label="Y column",
                                    placeholder="Choose Y…",
                                    data=[],
                                    w=240,
                                ),
                                dmc.Select(
                                    id="bivariate-color-select",
                                    label="Color by (optional)",
                                    placeholder="None",
                                    data=[],
                                    w=200,
                                    clearable=True,
                                ),
                            ],
                            align="flex-end",
                        ),
                        html.Div(
                            dmc.Text("Upload a dataset to explore bivariate relationships.", c="dimmed", ta="center", mt="xl"),
                            id="bivariate-content",
                        ),
                    ],
                    gap="md",
                ),
                value="bivariate",
                pt="md",
            ),
            # Multivariate
            dmc.TabsPanel(
                html.Div(
                    dmc.Text("Upload a dataset to view multivariate analysis.", c="dimmed", ta="center", mt="xl"),
                    id="multivariate-content",
                ),
                value="multivariate",
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
            # Preprocessing
            dmc.TabsPanel(
                dmc.Stack(
                    [
                        dmc.Accordion(
                            [
                                dmc.AccordionItem(
                                    [
                                        dmc.AccordionControl("Missing Value Imputation"),
                                        dmc.AccordionPanel(
                                            dmc.Stack(
                                                [
                                                    dmc.Group(
                                                        [
                                                            dmc.Select(
                                                                id="impute-strategy-select",
                                                                label="Strategy",
                                                                data=["mean", "median", "mode", "constant", "drop"],
                                                                value="mean",
                                                                w=200,
                                                            ),
                                                            dmc.TextInput(
                                                                id="impute-constant-value",
                                                                label="Constant value",
                                                                placeholder="0",
                                                                w=160,
                                                            ),
                                                            dmc.Button(
                                                                "Apply Imputation",
                                                                id="impute-apply-btn",
                                                                color="blue",
                                                                mt="lg",
                                                            ),
                                                        ],
                                                        align="flex-end",
                                                    ),
                                                    html.Div(id="impute-result"),
                                                ],
                                                gap="sm",
                                            ),
                                        ),
                                    ],
                                    value="imputation",
                                ),
                                dmc.AccordionItem(
                                    [
                                        dmc.AccordionControl("Outlier Treatment"),
                                        dmc.AccordionPanel(
                                            dmc.Stack(
                                                [
                                                    dmc.Group(
                                                        [
                                                            dmc.Select(
                                                                id="outlier-col-select",
                                                                label="Column",
                                                                placeholder="Choose column…",
                                                                data=[],
                                                                w=200,
                                                            ),
                                                            dmc.Select(
                                                                id="outlier-method-select",
                                                                label="Detection method",
                                                                data=["iqr", "zscore"],
                                                                value="iqr",
                                                                w=160,
                                                            ),
                                                            dmc.Select(
                                                                id="outlier-action-select",
                                                                label="Action",
                                                                data=["cap", "remove", "flag"],
                                                                value="cap",
                                                                w=140,
                                                            ),
                                                            dmc.Button(
                                                                "Apply",
                                                                id="outlier-apply-btn",
                                                                color="orange",
                                                                mt="lg",
                                                            ),
                                                        ],
                                                        align="flex-end",
                                                    ),
                                                    html.Div(id="outlier-treatment-result"),
                                                ],
                                                gap="sm",
                                            ),
                                        ),
                                    ],
                                    value="outliers",
                                ),
                                dmc.AccordionItem(
                                    [
                                        dmc.AccordionControl("Column Transformation"),
                                        dmc.AccordionPanel(
                                            dmc.Stack(
                                                [
                                                    dmc.Group(
                                                        [
                                                            dmc.Select(
                                                                id="transform-col-select",
                                                                label="Column",
                                                                placeholder="Choose column…",
                                                                data=[],
                                                                w=200,
                                                            ),
                                                            dmc.Select(
                                                                id="transform-method-select",
                                                                label="Transformation",
                                                                data=[
                                                                    "log1p", "sqrt", "square",
                                                                    "standardize", "normalize",
                                                                    "boxcox", "yeojohnson",
                                                                ],
                                                                value="log1p",
                                                                w=180,
                                                            ),
                                                            dmc.Button(
                                                                "Apply",
                                                                id="transform-apply-btn",
                                                                color="teal",
                                                                mt="lg",
                                                            ),
                                                        ],
                                                        align="flex-end",
                                                    ),
                                                    html.Div(id="transform-result"),
                                                ],
                                                gap="sm",
                                            ),
                                        ),
                                    ],
                                    value="transformation",
                                ),
                                dmc.AccordionItem(
                                    [
                                        dmc.AccordionControl("Categorical Encoding"),
                                        dmc.AccordionPanel(
                                            dmc.Stack(
                                                [
                                                    dmc.Group(
                                                        [
                                                            dmc.Select(
                                                                id="encode-col-select",
                                                                label="Column",
                                                                placeholder="Choose column…",
                                                                data=[],
                                                                w=200,
                                                            ),
                                                            dmc.Select(
                                                                id="encode-method-select",
                                                                label="Encoding",
                                                                data=["label", "onehot", "frequency"],
                                                                value="label",
                                                                w=160,
                                                            ),
                                                            dmc.Button(
                                                                "Apply",
                                                                id="encode-apply-btn",
                                                                color="violet",
                                                                mt="lg",
                                                            ),
                                                        ],
                                                        align="flex-end",
                                                    ),
                                                    html.Div(id="encode-result"),
                                                ],
                                                gap="sm",
                                            ),
                                        ),
                                    ],
                                    value="encoding",
                                ),
                            ],
                            multiple=True,
                            value=["imputation"],
                        ),
                        html.Div(id="preprocessing-summary"),
                    ],
                    gap="md",
                ),
                value="preprocessing",
                pt="md",
            ),
            # Features
            dmc.TabsPanel(
                dmc.Stack(
                    [
                        dmc.Group(
                            [
                                dmc.Select(
                                    id="feature-target-select",
                                    label="Target column",
                                    placeholder="Select target…",
                                    data=[],
                                    w=220,
                                ),
                                dmc.Select(
                                    id="feature-problem-type",
                                    label="Problem type",
                                    data=["regression", "classification"],
                                    value="regression",
                                    w=180,
                                ),
                                dmc.NumberInput(
                                    id="feature-k-select",
                                    label="Top K features",
                                    value=10,
                                    min=1,
                                    max=50,
                                    w=140,
                                ),
                                dmc.Button(
                                    "Analyse Features",
                                    id="feature-analyse-btn",
                                    color="blue",
                                    mt="lg",
                                ),
                            ],
                            align="flex-end",
                        ),
                        html.Div(
                            dmc.Text("Configure target column and click Analyse Features.", c="dimmed", ta="center", mt="xl"),
                            id="features-content",
                        ),
                    ],
                    gap="md",
                ),
                value="features",
                pt="md",
            ),
            # Modeling
            dmc.TabsPanel(
                dmc.Stack(
                    [
                        dmc.Paper(
                            dmc.Stack(
                                [
                                    dmc.Text("Experiment Design", fw=600, size="lg"),
                                    dmc.Divider(),
                                    dmc.Group(
                                        [
                                            dmc.Select(
                                                id="model-target-select",
                                                label="Target column",
                                                placeholder="Select target…",
                                                data=[],
                                                w=220,
                                            ),
                                            dmc.Select(
                                                id="model-problem-type",
                                                label="Problem type",
                                                data=["regression", "classification"],
                                                value="regression",
                                                w=180,
                                            ),
                                            dmc.Select(
                                                id="model-test-size",
                                                label="Test size",
                                                data=["0.1", "0.2", "0.25", "0.3"],
                                                value="0.2",
                                                w=120,
                                            ),
                                            dmc.Switch(
                                                id="model-scale-features",
                                                label="Scale features",
                                                checked=False,
                                                mt="lg",
                                            ),
                                        ],
                                        align="flex-end",
                                    ),
                                    dmc.MultiSelect(
                                        id="model-select",
                                        label="Models to train",
                                        placeholder="Select one or more models…",
                                        data=[],
                                        w="100%",
                                    ),
                                    dmc.Group(
                                        [
                                            dmc.Button(
                                                "Train & Evaluate",
                                                id="model-train-btn",
                                                color="blue",
                                                size="md",
                                            ),
                                            dmc.Button(
                                                "Compare All",
                                                id="model-compare-btn",
                                                color="teal",
                                                size="md",
                                                variant="outline",
                                            ),
                                        ],
                                        gap="sm",
                                    ),
                                ],
                                gap="sm",
                            ),
                            withBorder=True,
                            radius="md",
                            p="md",
                        ),
                        html.Div(id="model-loading-indicator"),
                        html.Div(
                            dmc.Text("Configure experiment and click Train & Evaluate.", c="dimmed", ta="center", mt="xl"),
                            id="modeling-content",
                        ),
                    ],
                    gap="md",
                ),
                value="modeling",
                pt="md",
            ),
            # Predictions
            dmc.TabsPanel(
                dmc.Stack(
                    [
                        dmc.Alert(
                            "Train a model first on the Modeling tab, then return here to generate predictions.",
                            title="ℹ️ Instructions",
                            color="blue",
                            id="predictions-alert",
                        ),
                        dmc.Group(
                            [
                                dmc.Button(
                                    "Generate Predictions on Test Set",
                                    id="predict-btn",
                                    color="blue",
                                ),
                                dmc.Button(
                                    "Download Model (.joblib)",
                                    id="model-download-btn",
                                    color="teal",
                                    variant="outline",
                                ),
                                dcc.Download(id="model-download"),
                            ],
                            gap="sm",
                        ),
                        html.Div(id="predictions-content"),
                    ],
                    gap="md",
                ),
                value="predictions",
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
        navbar={"width": 260, "breakpoint": "sm", "collapsed": {"mobile": True}},
        header={"height": 60},
        padding="0",
    )

    return dmc.MantineProvider(
        children=[
            dcc.Store(id="data-store", storage_type="memory"),
            dcc.Store(id="preprocessed-store", storage_type="memory"),
            dcc.Store(id="model-result-store", storage_type="memory"),
            app_shell,
        ],
        theme={
            "primaryColor": "blue",
            "fontFamily": "'Inter', sans-serif",
        },
        defaultColorScheme="light",
        id="mantine-provider",
    )


