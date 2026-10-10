#!/usr/bin/env python3
"""
Gringo Bandito Review Lab, Dash app.

Standalone:
    python -m app.app --port 8050

Mounted in flask_apps (after `make export_flask_apps FLASK_APPS=...`):
    from gb_reviews.app import create_app
    create_app(server=flask_server, url_base_pathname="/gb-reviews/")
"""

import argparse
from pathlib import Path

from dash import Dash, Input, Output

from . import data as data_mod
from .figures import heat_gap_figure
from .layout import (
    fans_children,
    foods_children,
    heat_table_for,
    hero_stats,
    kpi_tiles,
    page,
    rivals_children,
)

FONTS = (
    "https://fonts.googleapis.com/css2?family=Anton"
    "&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap"
)


def create_app(server=None, url_base_pathname: str = "/", data_dir: Path | None = None) -> Dash:
    data = data_mod.load(data_dir)
    kwargs = dict(
        title="Gringo Bandito Review Lab",
        assets_folder=str(Path(__file__).resolve().parent / "assets"),
        external_stylesheets=[FONTS],
        url_base_pathname=url_base_pathname,
    )
    if server is not None:
        kwargs["server"] = server
    app = Dash(__name__, **kwargs)
    app.layout = page(data)

    @app.callback(
        Output("kpis", "children"),
        Output("foods", "children"),
        Output("rivals", "children"),
        Output("fans", "children"),
        Input("sauce-filter", "value"),
    )
    def filter_scope(scope):
        return (
            kpi_tiles(data, scope),
            foods_children(data, scope),
            rivals_children(data, scope),
            fans_children(data, scope),
        )

    @app.callback(
        Output("heat-gap", "figure"),
        Output("hero-stats", "children"),
        Input("heat-view", "value"),
    )
    def switch_heat_view(view):
        return heat_gap_figure(heat_table_for(data, view)), hero_stats(data, view)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    create_app().run(port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
