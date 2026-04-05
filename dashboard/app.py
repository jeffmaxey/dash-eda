"""Dash application factory."""

from __future__ import annotations

import dash
from flask_caching import Cache

from dashboard.layout import create_layout
from dashboard.callbacks.data import register_data_callbacks
from dashboard.callbacks.charts import register_chart_callbacks
from dashboard.callbacks.preprocessing import register_preprocessing_callbacks
from dashboard.callbacks.features import register_features_callbacks
from dashboard.callbacks.modeling import register_modeling_callbacks


def create_app(server=None) -> dash.Dash:
    """Create and configure the Dash application.

    Parameters
    ----------
    server:
        Optional Flask server instance.  Useful for attaching to an
        existing WSGI server.

    Returns
    -------
    dash.Dash
        Fully configured application ready to run.
    """
    kwargs: dict = {
        "suppress_callback_exceptions": True,
        "use_pages": False,
        "title": "Dash EDA",
    }
    if server is not None:
        kwargs["server"] = server

    app = dash.Dash(__name__, **kwargs)

    # Flask-Caching (simple in-memory cache)
    cache = Cache(
        app.server,
        config={"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 300},
    )
    app._cache = cache  # keep a reference so tests can reach it

    app.layout = create_layout

    # Register callbacks
    register_data_callbacks(app)
    register_chart_callbacks(app)
    register_preprocessing_callbacks(app)
    register_features_callbacks(app)
    register_modeling_callbacks(app)

    return app
