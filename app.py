"""Root entry point for the Dash EDA application."""

from dashboard.app import create_app

app = create_app()
server = app.server

if __name__ == "__main__":
    app.run(debug=True, port=8050)
