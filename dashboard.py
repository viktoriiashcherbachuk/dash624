#!/usr/bin/env python3

from dash import Dash, html
import dash_bootstrap_components as dbc

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(
    [
        html.H1("Layout Prototype"),
        """
        I can do this!
        """,
    ]
)

if __name__ == "__main__":
    app.run_server(debug=False)
