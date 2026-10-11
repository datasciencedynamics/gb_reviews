"""
Gringo Bandito Review Lab dashboard.

Dash front end for the gb_reviews project: how every Amazon review of the
four sauces (2010 to 2023) talks about heat, flavor, vinegar and use, and the
headline finding that "Super Hot" is the sauce buyers most often call milder
than expected. Everything it shows comes from the flat files `make export_app`
writes to data/app in gb_reviews (copied to ./data next to this file in
flask_apps). The app needs no access to the pipeline, the labels or the API.

Two pages, same pattern as the bh_traffic dashboard:
    /            overview, plain language
    /technical   methods, statistics, validation

Run locally:
    python app/app.py --port 8050
"""

import argparse
import base64
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, no_update

HERE = Path(__file__).resolve().parent

## Customer report card PDF (report_card.py next to this file). Loaded by file
## path under a unique name so it cannot collide with other apps in the
## flask_apps launcher. Without reportlab the app still runs; the button hides.
try:
    _spec = importlib.util.spec_from_file_location(
        "gb_reviews_report_card", HERE / "report_card.py")
    report_card = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(report_card)
except (ImportError, FileNotFoundError) as _e:
    report_card = None
    print(f"gb_reviews: report card disabled ({_e}); pip install reportlab svglib")

################################################################################
# App settings
#
# Under flask_apps' DispatcherMiddleware the mount segment is stripped from
# PATH_INFO; requests_pathname_prefix tells the browser where to ask.
################################################################################
PREFIX = os.environ.get("GB_REVIEWS_URL_PREFIX", "/")
TITLE = "Gringo Bandito Review Lab"
AUTHOR = "Leon Shpaner, M.S., Maxime Boulat"
REPO = "https://github.com/datasciencedynamics/gb_reviews"
FONTS = (
    "https://fonts.googleapis.com/css2?family=Anton"
    "&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap"
)


def default_data_dir() -> Path:
    env = os.environ.get("GB_APP_DATA")
    if env:
        return Path(env)
    if (HERE / "data").is_dir():  # flask_apps layout
        return HERE / "data"
    return HERE.parent / "data" / "app"  # inside the gb_reviews repo


################################################################################
# Palette (dark ground, one orange accent, blue for complaints)
################################################################################
INK = "#F2EDE4"
INK2 = "#CFC7BB"
MUTED = "#A69E92"
FAINT = "#6B6157"
CARD = "#211B17"
LINE = "#3A322B"
LINE_SOFT = "#2B241F"
ACCENT = "#EE7A2B"
GOLD = "#F0B64A"
POS = "#EE7A2B"  # praise
NEG = "#6FA8DC"  # complaints (differs in lightness as well as hue)
NEUTRAL = "#8C8378"

SAUCES = ("original", "green", "super_hot", "spicy_yellow")
SAUCE_NAMES = {
    "all": "All sauces",
    "original": "Original Red",
    "green": "Green Habanero",
    "super_hot": "Super Hot",
    "spicy_yellow": "Spicy Yellow",
}
SAUCE_COLORS = {
    "original": "#D7392B",
    "green": "#4E9A55",
    "super_hot": "#EE7A2B",
    "spicy_yellow": "#E3B23C",
}
ASPECT_LABELS = {
    "heat": "Heat",
    "flavor": "Flavor",
    "vinegar": "Vinegar",
    "versatility": "Versatility",
    "value": "Value",
}
MIN_N_BAR = 20  # fewer heat reviews than this: no bar, just a note

## Data Science Dynamics logo, recolored for the dark ground. Embedded as a
## data URI so it never depends on a file being served.
DSD_LOGO_B64 = (
    "PHN2ZyB2aWV3Qm94PSIzMyA0MiAzODAgMTA0IiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAw"
    "MC9zdmciPgoKPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOTAsIDg1KSI+CiAgPGNpcmNsZSBjeD0i"
    "MCIgY3k9IjAiIHI9IjEzIiBmaWxsPSIjOERCOEU4Ii8+CiAgCiAgPGNpcmNsZSBjeD0iLTMwIiBj"
    "eT0iLTMwIiByPSI3IiBmaWxsPSIjOERCOEU4IiBvcGFjaXR5PSIwLjgiLz4KICA8Y2lyY2xlIGN4"
    "PSIyMCIgY3k9Ii0zNCIgcj0iNiIgZmlsbD0iIzhEQjhFOCIgb3BhY2l0eT0iMC42Ii8+CiAgPGNp"
    "cmNsZSBjeD0iLTUwIiBjeT0iLTEyIiByPSI1IiBmaWxsPSIjOERCOEU4IiBvcGFjaXR5PSIwLjQ1"
    "Ii8+CiAgCiAgPGNpcmNsZSBjeD0iLTMwIiBjeT0iMzAiIHI9IjciIGZpbGw9IiM4REI4RTgiIG9w"
    "YWNpdHk9IjAuOCIvPgogIDxjaXJjbGUgY3g9IjIwIiBjeT0iMzQiIHI9IjYiIGZpbGw9IiM4REI4"
    "RTgiIG9wYWNpdHk9IjAuNiIvPgogIDxjaXJjbGUgY3g9Ii01MCIgY3k9IjEyIiByPSI1IiBmaWxs"
    "PSIjOERCOEU4IiBvcGFjaXR5PSIwLjQ1Ii8+CiAgCiAgPGNpcmNsZSBjeD0iMzYiIGN5PSItMTAi"
    "IHI9IjQuNSIgZmlsbD0iIzhEQjhFOCIgb3BhY2l0eT0iMC40Ii8+CiAgPGNpcmNsZSBjeD0iMzYi"
    "IGN5PSIxMCIgcj0iNC41IiBmaWxsPSIjOERCOEU4IiBvcGFjaXR5PSIwLjQiLz4KICAKICA8bGlu"
    "ZSB4MT0iMCIgeTE9IjAiIHgyPSItMzAiIHkyPSItMzAiIHN0cm9rZT0iIzhEQjhFOCIgc3Ryb2tl"
    "LXdpZHRoPSIxLjUiIG9wYWNpdHk9IjAuNiIvPgogIDxsaW5lIHgxPSIwIiB5MT0iMCIgeDI9IjIw"
    "IiB5Mj0iLTM0IiBzdHJva2U9IiM4REI4RTgiIHN0cm9rZS13aWR0aD0iMS4yIiBvcGFjaXR5PSIw"
    "LjU1Ii8+CiAgPGxpbmUgeDE9IjAiIHkxPSIwIiB4Mj0iLTUwIiB5Mj0iLTEyIiBzdHJva2U9IiM4"
    "REI4RTgiIHN0cm9rZS13aWR0aD0iMSIgb3BhY2l0eT0iMC41Ii8+CiAgPGxpbmUgeDE9IjAiIHkx"
    "PSIwIiB4Mj0iLTMwIiB5Mj0iMzAiIHN0cm9rZT0iIzhEQjhFOCIgc3Ryb2tlLXdpZHRoPSIxLjUi"
    "IG9wYWNpdHk9IjAuNiIvPgogIDxsaW5lIHgxPSIwIiB5MT0iMCIgeDI9IjIwIiB5Mj0iMzQiIHN0"
    "cm9rZT0iIzhEQjhFOCIgc3Ryb2tlLXdpZHRoPSIxLjIiIG9wYWNpdHk9IjAuNTUiLz4KICA8bGlu"
    "ZSB4MT0iMCIgeTE9IjAiIHgyPSItNTAiIHkyPSIxMiIgc3Ryb2tlPSIjOERCOEU4IiBzdHJva2Ut"
    "d2lkdGg9IjEiIG9wYWNpdHk9IjAuNSIvPgogIDxsaW5lIHgxPSIwIiB5MT0iMCIgeDI9IjM2IiB5"
    "Mj0iLTEwIiBzdHJva2U9IiM4REI4RTgiIHN0cm9rZS13aWR0aD0iMSIgb3BhY2l0eT0iMC41NSIv"
    "PgogIDxsaW5lIHgxPSIwIiB5MT0iMCIgeDI9IjM2IiB5Mj0iMTAiIHN0cm9rZT0iIzhEQjhFOCIg"
    "c3Ryb2tlLXdpZHRoPSIxIiBvcGFjaXR5PSIwLjU1Ii8+CiAgCiAgPGxpbmUgeDE9Ii0zMCIgeTE9"
    "Ii0zMCIgeDI9IjIwIiB5Mj0iLTM0IiBzdHJva2U9IiM4REI4RTgiIHN0cm9rZS13aWR0aD0iMC43"
    "IiBvcGFjaXR5PSIwLjQiLz4KICA8bGluZSB4MT0iLTMwIiB5MT0iLTMwIiB4Mj0iLTUwIiB5Mj0i"
    "LTEyIiBzdHJva2U9IiM4REI4RTgiIHN0cm9rZS13aWR0aD0iMC43IiBvcGFjaXR5PSIwLjQiLz4K"
    "ICA8bGluZSB4MT0iLTMwIiB5MT0iMzAiIHgyPSIyMCIgeTI9IjM0IiBzdHJva2U9IiM4REI4RTgi"
    "IHN0cm9rZS13aWR0aD0iMC43IiBvcGFjaXR5PSIwLjQiLz4KICA8bGluZSB4MT0iLTMwIiB5MT0i"
    "MzAiIHgyPSItNTAiIHkyPSIxMiIgc3Ryb2tlPSIjOERCOEU4IiBzdHJva2Utd2lkdGg9IjAuNyIg"
    "b3BhY2l0eT0iMC40Ii8+CiAgPGxpbmUgeDE9IjM2IiB5MT0iLTEwIiB4Mj0iMjAiIHkyPSItMzQi"
    "IHN0cm9rZT0iIzhEQjhFOCIgc3Ryb2tlLXdpZHRoPSIwLjciIG9wYWNpdHk9IjAuMzUiLz4KICA8"
    "bGluZSB4MT0iMzYiIHkxPSIxMCIgeDI9IjIwIiB5Mj0iMzQiIHN0cm9rZT0iIzhEQjhFOCIgc3Ry"
    "b2tlLXdpZHRoPSIwLjciIG9wYWNpdHk9IjAuMzUiLz4KPC9nPgoKPGxpbmUgeDE9IjEyNiIgeTE9"
    "Ijc1IiB4Mj0iMTY2IiB5Mj0iNjYiIHN0cm9rZT0iIzhEQjhFOCIgc3Ryb2tlLXdpZHRoPSIxLjIi"
    "IG9wYWNpdHk9IjAuNTUiLz4KPGNpcmNsZSBjeD0iMTY2IiBjeT0iNjYiIHI9IjIuNSIgZmlsbD0i"
    "IzhEQjhFOCIgb3BhY2l0eT0iMC41NSIvPgoKPHBhdGggZD0iTTEwMyw4OCBMMTEyLDg3IEwxMTks"
    "NzggTDEyNiw5NCBMMTMzLDc0IEwxNDAsOTAgTDE0Nyw3NiBMMTU0LDg4IEwxNTksNzQgTDE2Niw4"
    "NiIgCiAgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjOERCOEU4IiBzdHJva2Utd2lkdGg9IjEuOCIgb3Bh"
    "Y2l0eT0iMC42NSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5k"
    "Ii8+CjxjaXJjbGUgY3g9IjE2NiIgY3k9Ijg2IiByPSIzIiBmaWxsPSIjOERCOEU4IiBvcGFjaXR5"
    "PSIwLjY1Ii8+Cgo8bGluZSB4MT0iMTI2IiB5MT0iOTUiIHgyPSIxNjYiIHkyPSIxMDYiIHN0cm9r"
    "ZT0iIzhEQjhFOCIgc3Ryb2tlLXdpZHRoPSIxLjIiIG9wYWNpdHk9IjAuNTUiLz4KPGNpcmNsZSBj"
    "eD0iMTY2IiBjeT0iMTA2IiByPSIyLjUiIGZpbGw9IiM4REI4RTgiIG9wYWNpdHk9IjAuNTUiLz4K"
    "Cjx0ZXh0IHg9IjE3MiIgeT0iNjgiIGZvbnQtZmFtaWx5PSJIZWx2ZXRpY2EsIEFyaWFsLCBzYW5z"
    "LXNlcmlmIiBmb250LXNpemU9IjM0IiBmb250LXdlaWdodD0iNzAwIiBmaWxsPSIjOERCOEU4IiBs"
    "ZXR0ZXItc3BhY2luZz0iNSI+REFUQTwvdGV4dD4KPHRleHQgeD0iMTcyIiB5PSI5OCIgZm9udC1m"
    "YW1pbHk9IkhlbHZldGljYSwgQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMzQiIGZvbnQt"
    "d2VpZ2h0PSI3MDAiIGZpbGw9IiM4REI4RTgiIGxldHRlci1zcGFjaW5nPSI1Ij5TQ0lFTkNFPC90"
    "ZXh0Pgo8dGV4dCB4PSIxNzIiIHk9IjEyOCIgZm9udC1mYW1pbHk9IkhlbHZldGljYSwgQXJpYWws"
    "IHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMzQiIGZvbnQtd2VpZ2h0PSI3MDAiIGZpbGw9IiM4REI4"
    "RTgiIGxldHRlci1zcGFjaW5nPSI1Ij5EWU5BTUlDUzwvdGV4dD4KCjwvc3ZnPg=="
)


################################################################################
# Data
################################################################################
def _json(p: Path) -> dict:
    return json.loads(p.read_text()) if p.is_file() else {}


def _pq(p: Path) -> pd.DataFrame:
    return pd.read_parquet(p) if p.is_file() else pd.DataFrame()


def load_data(data_dir=None) -> SimpleNamespace:
    d = Path(data_dir) if data_dir else default_data_dir()
    data = SimpleNamespace(
        summary=_json(d / "summary.json"),
        changepoint=_json(d / "changepoint.json"),
        aspects=_pq(d / "aspects.parquet"),
        heat=_pq(d / "heat.parquet"),
        heat_single=_pq(d / "heat_single.parquet"),
        foods=_pq(d / "foods.parquet"),
        competitors=_pq(d / "competitors.parquet"),
        fans=_pq(d / "fans.parquet"),
    )
    data.ready = bool(data.summary)
    data.scope_all = data.summary.get("scopes", {}).get("all", {})
    data.validation = data.summary.get("validation", {})
    return data


################################################################################
# Helpers
################################################################################
def _ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and np.isnan(x))


def pct(x, digits: int = 0) -> str:
    return f"{x:.{digits}%}" if _ok(x) else "—"


def num(x, fmt: str = "{:,.0f}") -> str:
    return fmt.format(x) if _ok(x) else "—"


def fmt_p(p) -> str:
    if not _ok(p):
        return "—"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"

def model_name(model_id):
    """'claude-haiku-5-5' -> 'Claude Haiku 5.5'."""
    if not model_id:
        return None
    parts = model_id.split("-")
    if parts[0] == "claude" and len(parts) >= 4:
        return f"Claude {parts[1].title()} {'.'.join(parts[2:])}"
    return model_id

def kappa_band(k) -> str:
    if not _ok(k):
        return "n/a"
    for cut, name in ((0.8, "almost perfect"), (0.6, "substantial"),
                      (0.4, "moderate"), (0.2, "fair"), (0.0, "slight")):
        if k > cut:
            return name
    return "poor"


def years(data) -> tuple:
    y0 = (data.scope_all.get("date_min") or "")[:4]
    y1 = (data.scope_all.get("date_max") or "")[:4]
    return y0, y1


def dot(sauce: str):
    return html.Span(className="dot", style={"background": SAUCE_COLORS[sauce]},
                     **{"aria-hidden": "true"})


def stack_bar(parts):
    """Horizontal 100% bar from [(share, color, label), ...]."""
    return html.Div(className="stackbar", children=[
        html.Div(style={"width": f"{max(0, s) * 100:.1f}%", "background": c},
                 title=f"{lab}: {s:.0%}")
        for s, c, lab in parts if _ok(s)
    ])


def legend(items):
    return html.Div(className="legend", children=[
        html.Span(className="legend-item", children=[
            html.Span(className="swatch", style={"background": c}), lab])
        for c, lab in items
    ])


################################################################################
# Figures
################################################################################
def _base(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=16, t=8, b=48),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", color=MUTED, size=13),
        hoverlabel=dict(bgcolor=CARD, bordercolor=LINE, font_color=INK),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE, tickcolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zeroline=False)
    return fig


def empty_figure(message: str, height: int = 300) -> go.Figure:
    fig = _base(go.Figure(), height)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.add_annotation(text=message, showarrow=False, font=dict(size=15, color=FAINT))
    return fig

def heat_gap_figure(heat: pd.DataFrame, show_ci: bool = True, plain: bool = False) -> go.Figure:
    """Share of heat reviews calling each sauce milder than expected, with
    Wilson 95% intervals. Value and n sit in the axis label so nothing collides
    with the error bars and the chart keeps its width on a phone.
    plain=True (overview): drops sauces with too few heat reviews and writes
    "26 reviews" instead of "n=26"."""
    if heat.empty:
        return empty_figure("Data pending")
    pad = "\u00a0\u00a0\u00a0"  # non-breaking spaces: gap between labels and bars
    t = heat.set_index("sauce").reindex(["original", "green", "spicy_yellow", "super_hot"])
    ticks, rates, lo, hi, colors = [], [], [], [], []
    for s, r in t.iterrows():
        n = int(r["n_heat"]) if pd.notna(r["n_heat"]) else 0
        if n >= MIN_N_BAR:
            count = f"{n} reviews" if plain else f"n={n}"
            ticks.append(f"<b>{SAUCE_NAMES[s]}</b>{pad}<br>"
                         f"{r['too_mild']:.0%} · {count}{pad}")
            rates.append(r["too_mild"])
            lo.append(r["too_mild"] - r["too_mild_lo"])
            hi.append(r["too_mild_hi"] - r["too_mild"])
        elif plain:
            continue
        else:
            ticks.append(f"<b>{SAUCE_NAMES[s]}</b>{pad}<br>too few (n={n}){pad}")
            rates.append(0)
            lo.append(0)
            hi.append(0)
        colors.append(ACCENT if s == "super_hot" else NEUTRAL)

    upper = max((r + h for r, h in zip(rates, hi)), default=0.5)
    fig = go.Figure(go.Bar(
        y=ticks, x=rates, orientation="h", marker=dict(color=colors),
        error_x=dict(type="data", visible=show_ci, symmetric=False, array=hi, arrayminus=lo,
                     color=INK, thickness=1.5, width=6),
        hovertemplate="%{x:.0%} milder than expected<extra></extra>",
    ))
    fig = _base(fig, 320 if len(ticks) > 3 else 260)
    fig.update_layout(bargap=0.35)
    fig.update_xaxes(tickformat=".0%", range=[0, min(1, upper * 1.1 + 0.02)],
                     gridcolor=LINE, showgrid=True, nticks=6,
                     title_text="Milder than expected", title_font=dict(size=12))
    fig.update_yaxes(showgrid=False, automargin=True,
                     tickfont=dict(color=INK, size=13, family="IBM Plex Mono, monospace"))
    return fig

def changepoint_figure(cp: dict, mark_split: bool = True, height: int = 360) -> go.Figure:
    """Trailing six-month share of Super Hot heat reviews calling it milder.
    mark_split adds the best-fitting split and its bootstrap band; the overview
    leaves it off, because a marked split reads as a change even when the test
    finds none."""
    series = pd.DataFrame(cp.get("series") or [])
    if series.empty:
        return empty_figure("Data pending", height)
    series["month"] = pd.to_datetime(series["month"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.concat([series["month"], series["month"][::-1]]),
        y=pd.concat([series["hi"], series["lo"][::-1]]),
        fill="toself", fillcolor="rgba(238,122,43,0.18)", line=dict(width=0),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=series["month"], y=series["rate"], mode="lines",
        line=dict(color=ACCENT, width=2.5), customdata=series["n"],
        hovertemplate="%{x|%b %Y}<br>%{y:.0%} milder than expected<br>"
                      "%{customdata} heat reviews in window<extra></extra>",
    ))
    if mark_split and cp.get("status") == "ok":
        ci = cp.get("change_date_ci")
        if ci:
            fig.add_vrect(x0=ci[0], x1=ci[1], fillcolor=GOLD, opacity=0.08, line_width=0)
        fig.add_vline(x=cp["change_date"], line=dict(color=GOLD, width=2, dash="dash"))
        sig = "" if cp["p_value"] < 0.05 else ", not significant"
        fig.add_annotation(
            x=cp["change_date"], y=1, yref="paper", xanchor="right", yanchor="top",
            text=f"Best-fitting split {pd.to_datetime(cp['change_date']):%b %Y}{sig}  ",
            showarrow=False, font=dict(color=GOLD, size=13),
        )
    fig = _base(fig, height)
    fig.update_layout(margin=dict(l=56, r=16, t=8, b=40))
    fig.update_yaxes(tickformat=".0%", rangemode="tozero",
                     title_text="Milder than expected", title_font=dict(size=12))
    return fig



def graph(fig, **kw):
    """dcc.Graph with an explicit height taken from the figure. Older Dash
    versions give a responsive graph's container no height of its own, so
    the chart overflows onto the content below it."""
    return dcc.Graph(figure=fig, style={"height": f"{fig.layout.height or 320}px"},
                     config={"displayModeBar": False, "responsive": True}, **kw)

################################################################################
# Shared chrome: header, nav, section heads, footer
################################################################################
OVERVIEW_SECTIONS = [
    ("sec-heat", "Super Hot Fans"),
    ("sec-time", "Over Time"),
    ("sec-sauces", "The Four Sauces"),
    ("sec-aspects", "What People Say"),
    ("sec-uses", "Uses & Rivals"),
    ("sec-fans", "The Fans"),
    ("sec-howto", "How to Read"),
]

TECH_SECTIONS = [
    ("tech-heat", "Heat Comparison"),
    ("tech-time", "Over Time"),
    ("tech-validation", "Label Validation"),
    ("tech-aspects", "Aspect Detail"),
    ("tech-scoville", "Scoville"),
    ("tech-methods", "Methods"),
    ("tech-data", "Data Notes"),
]

def logo_link():
    return html.A(
        href="https://www.datasciencedynamics.com", title="Data Science Dynamics",
        className="logo-link",
        children=html.Div(role="img", title="Data Science Dynamics", className="dsd-logo",
                          style={"backgroundImage":
                                 f"url(data:image/svg+xml;base64,{DSD_LOGO_B64})"}),
    )


def kicker(app: Dash, text: str):
    return html.Div(className="kicker-row", children=[
        html.Img(src=app.get_asset_url("gb_reviews_icon.svg"), alt="", className="kicker-icon"),
        html.Div(text, className="eyebrow"),
    ])


def kpi_tiles(data):
    s = data.scope_all
    y0, y1 = years(data)
    tiles = [
        ("Reviews analyzed", num(s.get("n")), f"{pct(s.get('verified_share'))} verified purchases"),
        ("Date range", f"{y0}\u2013{y1}" if y0 else "\u2014", "Amazon Reviews\u201923 release"),
        ("Average rating", num(s.get("avg_rating"), "{:.2f}"), "Out of 5 stars"),
        ("Discuss heat", pct(s.get("heat_mention_share")), "Share of reviews"),
    ]
    return html.Section(className="tiles", **{"aria-label": "Summary"}, children=[
        html.Div(className="tile", children=[
            html.Div(label, className="muted small"),
            html.Div(value, className="mono tile-value"),
            html.Div(note, className="muted small"),
        ])
        for label, value, note in tiles
    ])


def overview_header(app: Dash, data):
    y0, y1 = years(data)
    span = (f"{int(y1) - int(y0) + 1} years of"
            if y0.isdigit() and y1.isdigit() else "")
    return html.Header(className="header", children=[
        html.Div(className="header-text", children=[
            kicker(app, "Hot sauce · Gringo Bandito"),
            html.H1(f"What {span} buyers say about the sauce".replace("  ", " "),
                    className="h1"),
            html.Div(AUTHOR, className="byline"),
            html.P(
                "We read every Amazon review of Gringo Bandito's four sauces and scored "
                "what each one says about heat, flavor, vinegar, value and how people use "
                "it. Buyers love the flavor, and when they compare it with the big names, "
                "they mostly pick Gringo Bandito. The one thing they keep asking for: more "
                "heat in Super Hot.",
                className="lede",
            ),
        ]),
        logo_link(),
    ])


def tech_header(app: Dash):
    return html.Header(className="header", children=[
        html.Div(className="header-text", children=[
            kicker(app, "Gringo Bandito Review Lab · Technical details"),
            html.H1("Methods, statistics and validation", className="h1"),
            html.Div(AUTHOR, className="byline"),
            html.P(
                "Everything behind the overview: where the reviews come from, how each "
                "review was labeled and how those labels were checked, the heat comparison "
                "and its tests, and the search for a change over time.",
                className="lede",
            ),
        ]),
        logo_link(),
    ])


def nav_label():
    return html.Span("Jump to", className="nav-label")


def report_button():
    # Same ids on both pages; only one page is rendered at a time.
    return [
        *([html.A("⤓  Report card", id="dl-report", n_clicks=0,
                  className="nav-btn", title="One-page PDF of what customers say")]
          if report_card else []),
        dcc.Download(id="dl-report-file"),
    ]


def overview_nav(tech_path: str):
    return html.Nav(className="page-nav", **{"aria-label": "Sections"}, children=[
        nav_label(),
        *[html.A(label, href=f"#{a}", className="nav-btn") for a, label in OVERVIEW_SECTIONS],
        dcc.Link("Technical details", href=tech_path, className="nav-btn nav-strong"),
        *report_button(),
    ])


def tech_nav(home: str):
    return html.Nav(className="page-nav", **{"aria-label": "Sections"}, children=[
        dcc.Link("Back to overview", href=home, className="nav-btn nav-strong"),
        nav_label(),
        *[html.A(label, href=f"#{a}", className="nav-btn") for a, label in TECH_SECTIONS],
        *report_button(),
    ])


def section_head(text: str, anchor: str, tech_path: str | None = None,
                 detail: str | None = None, size: str = "lg"):
    children = [html.H2(text, id=anchor, className=f"h2 h2-{size} anchor")]
    if detail and tech_path:
        children.append(dcc.Link("Technical details →", href=f"{tech_path}#{detail}",
                                 className="details-link"))
    return html.Div(className="section-head", children=children)


def footer(data, tech_path: str):
    y1 = years(data)[1]
    return html.Footer(className="footer", children=[
        html.P([
            "Source: public Amazon customer reviews, via McAuley Lab's Amazon Reviews’23 "
            "research dataset. ",
            data.summary.get("citation", ""),
            " An independent analysis; not affiliated with or endorsed by Gringo Bandito "
            "or Dexter Holland.",
        ], className="note"),
        html.Div(className="footer-links", children=[
            f"© {pd.Timestamp.today().year} ",
            html.A("Data Science Dynamics", href="https://www.datasciencedynamics.com"),
            html.Span(" · ", className="sep"), AUTHOR,
            html.Span(" · ", className="sep"),
            dcc.Link("Technical details", href=tech_path),
            html.Span(" · ", className="sep"),
            html.A("GitHub Repo", href=REPO, target="_blank", rel="noopener noreferrer"),
        ]),
    ])


################################################################################
# Overview sections
################################################################################
def heat_story(data) -> str:
    g = data.summary.get("heat_gap", {}).get("single", {})
    if not g:
        return "Results appear here once the pipeline has run."
    text = (f"About {g['rate_a']:.0%} of buyers who talked about Super Hot's heat wanted "
            f"more of it, compared with {g['rate_b']:.0%} for Original Red.")
    st = data.summary.get("super_hot_stars", {})
    mild, exp = st.get("too_mild"), st.get("as_expected")
    if mild and exp:
        text += (f" They still liked it, at {mild['mean']:.1f} stars on average, but buyers "
                 f"who got the heat they expected gave it {exp['mean']:.1f}. Living up to the "
                 "name is worth about a star.")
    return text

def overview_heat(data, tech_path):
    return html.Section(className="card hero", children=[
        section_head("Super Hot fans want it hotter", "sec-heat", tech_path,
                     "tech-heat", "xl"),
        html.P(
            "Of the reviews that talk about heat, the share saying the sauce was milder "
            "than they expected. Original Red and Green Habanero mostly deliver what people "
            "expect. Super Hot's buyers are asking for more.",
            className="body",
        ),
        dcc.Graph(figure=heat_gap_figure(data.heat_single, show_ci=False, plain=True)),
        html.P(heat_story(data), className="callout"),
        html.P("Percentages count only reviews about a single sauce, so each verdict clearly "
               "belongs to it. Star averages use every Super Hot review that mentions heat.",
               className="note"),
    ])

def overview_time(data, tech_path):
    cp = data.changepoint
    if cp.get("status") == "ok" and cp.get("p_value", 0) >= 0.05:
        reading = ("Yes. The share bounces around from year to year, but a statistical test "
                   "finds no lasting shift, and the swings at the edges rest on only a handful "
                   "of reviews. Super Hot has read as milder than its name for as long as "
                   "there are reviews to say so.")
    elif cp.get("status") == "ok":
        reading = (f"No. The share shifted around "
                   f"{pd.to_datetime(cp['change_date']):%B %Y}, and the shift is larger than "
                   "chance would explain.")
    else:
        reading = "Results appear here once the pipeline has run."
    return html.Section(className="card hero", children=[
        section_head("Has it always been this way?", "sec-time", tech_path, "tech-time"),
        html.P("Among Super Hot reviews that talk about heat, the share saying it was milder "
               "than expected, over time. Each point covers the six months before it; the "
               "shaded ribbon is the range the true share likely falls in.",
               className="body"),
        graph(changepoint_figure(cp, mark_split=False)),
        html.P(reading, className="callout"),
    ])


def _aspect_extremes(data, sauce):
    a = data.aspects
    if a.empty:
        return None, None
    sub = a[(a["sauce"] == sauce) & a["net"].notna()]
    if sub.empty:
        return None, None
    best = sub.loc[sub["net"].idxmax()]
    worst = sub.loc[sub["net"].idxmin()]
    return (ASPECT_LABELS[best["aspect"]] if best["net"] > 0.3 else None,
            ASPECT_LABELS[worst["aspect"]] if worst["net"] < 0 else None)


def overview_sauces(data, tech_path):
    h = data.heat.set_index("sauce") if not data.heat.empty else pd.DataFrame()
    cards = []
    for s in SAUCES:
        r = h.loc[s] if s in h.index else None
        n_heat = int(r["n_heat"]) if r is not None else 0
        shu = r["labeled_shu"] if r is not None else None
        praised, complained = _aspect_extremes(data, s)
        if r is not None and n_heat >= MIN_N_BAR:
            felt = [
                stack_bar([(r["too_mild"], NEG, "milder"),
                           (r["as_expected"], NEUTRAL, "as expected"),
                           (r["too_hot"], POS, "hotter")]),
                html.Div(f"{r['too_mild']:.0%} milder · {r['as_expected']:.0%} as "
                         f"expected · {r['too_hot']:.0%} hotter", className="note tight"),
            ]
        else:
            felt = [html.Div("Too few reviews about its heat.", className="note tight")]
        cards.append(html.Div(className="card sauce-card", children=[
            html.Div(className="sauce-name", children=[dot(s), SAUCE_NAMES[s]]),
            html.Div(className="sauce-shu", children=[
                html.Span(num(shu), className="mono big"),
                html.Span(" Scoville on the label", className="muted small"),
            ]),
            html.Div("How hot it felt to reviewers", className="label"),
            *felt,
            html.Div(className="sauce-tags", children=[
                html.Span(["Most praised: ", html.B(praised)], className="tag")
                if praised else None,
                html.Span(["Most complaints: ", html.B(complained)], className="tag tag-neg")
                if complained else None,
            ]),
        ]))
    return html.Section(className="stack-lg", children=[
        section_head("The four sauces", "sec-sauces", tech_path, "tech-scoville"),
        html.P("What each bottle promises on the label, next to how hot reviewers say it "
               "felt. Spicy Yellow is the newest sauce, so it has the fewest reviews.",
               className="muted"),
        legend([(NEG, "Milder than expected"), (NEUTRAL, "As expected"),
                (POS, "Hotter than expected")]),
        html.Div(className="cards", children=cards),
    ])


def _aspect_row(r):
    label = ASPECT_LABELS[r["aspect"]]
    if not _ok(r.get("net")):
        return html.Div(className="aspect", children=[
            html.Div(className="row-between small", children=[
                html.Span(label), html.Span("too few mentions", className="muted")]),
            stack_bar([]),
        ])
    neg = r["neg_share"]
    pos = r["net"] + neg
    mixed = max(0.0, 1 - pos - neg)
    return html.Div(className="aspect", children=[
        html.Div(className="row-between small", children=[
            html.Span(label),
            html.Span(f"{pos:.0%} praise · in {r['mention_share']:.0%} of reviews",
                      className="mono muted"),
        ]),
        stack_bar([(pos, POS, "praise"), (mixed, NEUTRAL, "mixed"), (neg, NEG, "complaints")]),
    ])


def overview_aspects(data, tech_path):
    a = data.aspects
    cards = []
    for s in SAUCES:
        sub = a[a["sauce"] == s] if not a.empty else pd.DataFrame()
        n = int(sub["n_reviews"].iat[0]) if not sub.empty else None
        rows = ([_aspect_row(r) for _, r in sub.iterrows()] if not sub.empty else
                [_aspect_row({"aspect": k, "net": None}) for k in ASPECT_LABELS])
        cards.append(html.Div(className="card aspect-card", children=[
            html.Div(className="sauce-name", children=[dot(s), SAUCE_NAMES[s]]),
            html.Div(f"{num(n)} reviews", className="muted small"),
            *rows,
        ]))
    return html.Section(className="stack-lg", children=[
        section_head("What people say about each sauce", "sec-aspects", tech_path,
                     "tech-aspects"),
        html.P("For each topic, how reviews that bring it up lean, and how often it comes "
               "up at all.", className="muted"),
        legend([(POS, "Praise"), (NEUTRAL, "Mixed"), (NEG, "Complaints")]),
        html.Div(className="cards", children=cards),
    ])


def foods_children(data, scope):
    f = data.foods
    f = f[f["scope"] == scope] if not f.empty else f
    if f.empty:
        return [html.Span("No food mentions for this sauce yet.", className="muted small")]
    top = f["n"].max()
    return [
        html.Span(className="chip", style={"fontSize": f"{13 + 9 * r['n'] / top:.0f}px"},
                  children=[r["food"], " ", html.Span(num(r["n"]), className="mono muted")])
        for _, r in f.iterrows()
    ]


def rivals_children(data, scope):
    c = data.competitors
    c = c[c["scope"] == scope] if not c.empty else c
    if c.empty:
        return [html.Span("No other brands mentioned for this sauce.", className="muted small")]
    rows = []
    for _, r in c.iterrows():
        n = r["n"]
        rows.append(html.Div(className="rival", children=[
            html.Span(r["brand"], className="rival-name"),
            stack_bar([(r["gb_better"] / n, POS, "Gringo Bandito preferred"),
                       (r["similar"] / n, NEUTRAL, "about the same"),
                       (r["gb_worse"] / n, NEG, "other brand preferred"),
                       (r["unclear"] / n, LINE, "unclear")]),
            html.Span(f"{n} mentions", className="mono muted small rival-n"),
        ]))
    return rows


def fans_children(data, scope):
    f = data.fans
    f = f[f["scope"] == scope] if not f.empty else f
    if f.empty:
        return [html.Span("Pending.", className="muted small")]
    out = []
    for g in f.itertuples():
        lines = [("Reviews", num(g.n)), ("Average stars", num(g.avg_rating, "{:.2f}")),
                 ("Say they bought it again", pct(g.reorder_share))]
        if scope == "all":
            lines.append(("Sauce they talk about most", g.top_sauce or "—"))
        out.append(html.Div(className="fan-card", children=[
            html.Div(g.group, className="strong"),
            *[html.Div(className="row-between small", children=[
                html.Span(k), html.Span(v, className="mono")]) for k, v in lines],
        ]))
    return out


def overview_uses(data):
    return html.Section(className="stack-lg", children=[
        section_head("How people use it, and what they compare it to", "sec-uses"),
        html.Div(className="filter-row", children=[
            html.Span("Show for", className="label"),
            dcc.RadioItems(
                id="sauce-filter",
                options=[{"label": SAUCE_NAMES[k], "value": k} for k in ("all",) + SAUCES],
                value="all", className="pills", inputClassName="pill-input",
                labelClassName="pill",
            ),
        ]),
        html.Div(className="two-up", children=[
            html.Div(className="card stack", children=[
                html.H3("What it goes on", className="h3"),
                html.P("Foods named in reviews, sized by how often they come up.",
                       className="muted small"),
                html.Div(id="foods", className="chips",
                         children=foods_children(data, "all")),
            ]),
            html.Div(className="card stack", children=[
                html.H3("Compared against", className="h3"),
                html.P("Other brands reviewers bring up, and which one they prefer.",
                       className="muted small"),
                legend([(POS, "Gringo Bandito"), (NEUTRAL, "About the same"),
                        (NEG, "The other brand")]),
                html.Div(id="rivals", className="stack",
                         children=rivals_children(data, "all")),
            ]),
        ]),
        html.Div(className="card stack", children=[
            html.H3("Came for the band, stayed for the sauce?", id="sec-fans",
                    className="h3 anchor"),
            html.P("Reviewers who mention The Offspring or Dexter Holland, compared with "
                   "everyone else.", className="muted small"),
            html.Div(id="fans", className="fans", children=fans_children(data, "all")),
        ]),
    ])


def overview_howto(data, tech_path):
    y0, y1 = years(data)
    span = f"from {y0} to {y1}" if y0 else ""
    items = [
        ("Milder than expected. ",
         "A review counts here when it talks about how hot the sauce is and says it was "
         "less hot than the reviewer wanted or expected, even if they liked it."),
        ("Reviews about one sauce. ",
         "Many reviews are for variety packs and cover several sauces at once. The Super "
         "Hot comparison uses only reviews about a single sauce, so each verdict clearly "
         "belongs to it."),
        ("Praise and complaints. ",
         "For each topic, the share of reviews mentioning it that are positive, mixed or "
         "negative. Topics mentioned in fewer than five reviews are left blank."),
        ("Keep in mind. ",
         f"These are Amazon reviews {span}, written by people who chose to "
         "write one. Each review was read and scored by an AI model; a second model "
         "rechecked a sample and agreed closely, and the scores line up with the star "
         "ratings. The details are under "),
    ]
    paras = [html.P([html.B(k), v], className="howto") for k, v in items]
    paras[-1].children.extend([dcc.Link("Technical details", href=tech_path), "."])
    return html.Section(className="card stack", children=[
        html.H2("How to read this page", id="sec-howto", className="h2 h2-md anchor"),
        *paras,
    ])


def overview_layout(app, data, home, tech_path):
    banner = [] if data.ready else [html.Div(
        "No pipeline output found. Run `make full_pipeline` (or set GB_APP_DATA) to fill "
        "this page.", className="banner")]
    return html.Div(className="page-body", children=[
        *banner,
        overview_header(app, data),
        overview_nav(tech_path),
        kpi_tiles(data),
        overview_heat(data, tech_path),
        overview_time(data, tech_path),
        overview_sauces(data, tech_path),
        overview_aspects(data, tech_path),
        overview_uses(data),
        overview_howto(data, tech_path),
        footer(data, tech_path),
    ])


################################################################################
# Technical sections
################################################################################
def tech_table(columns, rows, numeric=()):
    return html.Div(className="table-wrap", children=html.Table(className="table", children=[
        html.Thead(html.Tr([html.Th(c, scope="col") for c in columns])),
        html.Tbody([html.Tr([html.Td(v, className="mono" if c in numeric else None)
                             for c, v in zip(columns, row)]) for row in rows]),
    ]))


def tech_section(anchor, title, children):
    return html.Section(className="card stack", children=[
        html.H2(title, id=anchor, className="h2 h2-md anchor"), *children])


def tech_methods(data):
    v = data.validation
    y0, y1 = years(data)
    s = data.scope_all
    return tech_section("tech-methods", "Methods", [
        html.Ul(className="methods-list", children=[
            html.Li([html.B("Data. "),
                     f"{num(s.get('n'))} Amazon reviews of Gringo Bandito listings in the "
                     f"Grocery and Gourmet Food category of McAuley Lab's Amazon Reviews’23 "
                     f"release, {y0} to {y1}; {pct(s.get('verified_share'), 1)} verified "
                     "purchases. Duplicate and empty reviews removed."]),
            html.Li([html.B("Sauce attribution. "),
                     "Most reviews sit on one parent listing that pools several flavor and "
                     "variety-pack variations. Each child product ID was matched to its "
                     "flavor by hand; variety packs and unmatched IDs get their sauce from the "
                     "review text instead (the sauces the review discusses)."]),
            html.Li([html.B("Labeling models. "),
                     f"Every review was labeled by Anthropic's "
                     f"{model_name(v.get('primary_model')) or 'Claude'} through the Anthropic "
                     "API, one review per request, filling a fixed schema through a forced "
                     "tool call: sentiment for heat, flavor, vinegar, versatility and value; "
                     "heat direction (milder than expected, as expected, hotter); sauces "
                     "discussed; foods; competitor comparisons; and mentions of a recipe "
                     "change, The Offspring, and reordering. A random 10% was relabeled "
                     f"independently by {model_name(v.get('check_model')) or 'a second model'} "
                     "to measure agreement (see Label validation). The prompt and schema are "
                     "in the repository (labeling/schema.py). Labels come from a single pass "
                     "at the API's default sampling settings, so a rerun can differ slightly "
                     "on borderline reviews."]),
            html.Li([html.B("Statistics. "),
                     "Shares carry Wilson 95% intervals. Sauce comparisons use Fisher's exact "
                     "test. The change over time is a single change point in a Bernoulli rate, "
                     "tested by permutation and dated by bootstrap. Cells built on fewer than "
                     "five reviews are suppressed."]),
        ]),
    ])


def heat_stats(data, view):
    g = data.summary.get("heat_gap", {}).get(view, {})
    if not g:
        return [html.P("Results appear here once the pipeline has run.", className="muted small")]
    return [html.Div(className="hero-stats", children=[
        html.Div(className="hero-stat", children=[
            html.Div("Super Hot", className="muted small"),
            html.Div(f"{g['rate_a']:.1%}", className="mono"),
            html.Div(f"n = {g['n_a']}", className="muted small")]),
        html.Div(className="hero-stat", children=[
            html.Div("Original Red", className="muted small"),
            html.Div(f"{g['rate_b']:.1%}", className="mono"),
            html.Div(f"n = {g['n_b']}", className="muted small")]),
        html.Div(className="hero-stat", children=[
            html.Div("Difference", className="muted small"),
            html.Div(f"{g['diff'] * 100:+.1f} points", className="mono"),
            html.Div(f"Fisher exact {fmt_p(g['fisher_p'])}", className="muted small")]),
    ])]


def heat_detail_table(data):
    rows = []
    for view, label, t in (("single", "One sauce", data.heat_single),
                           ("all", "All heat reviews", data.heat)):
        if t.empty:
            continue
        for _, r in t.iterrows():
            ok = _ok(r.get("too_mild"))
            rows.append([
                label, SAUCE_NAMES[r["sauce"]], num(r["n_heat"]),
                num(r.get("k_too_mild")),
                pct(r["too_mild"], 1) if ok else "—",
                f"{r['too_mild_lo']:.1%} to {r['too_mild_hi']:.1%}" if ok else "—",
                pct(r["too_hot"], 1) if ok else "—",
            ])
    if not rows:
        return html.P("Pending.", className="muted small")
    return tech_table(["Reviews", "Sauce", "Heat reviews", "Milder (k)", "Milder",
                       "95% CI (Wilson)", "Hotter"], rows,
                      numeric=("Heat reviews", "Milder (k)", "Milder", "95% CI (Wilson)", "Hotter"))


def tech_heat(data):
    return tech_section("tech-heat", "Heat comparison", [
        html.P("Outcome: heat_direction = too_mild among reviews whose heat_direction is not "
               "“not mentioned”. Heat direction is labeled once per review, so a "
               "variety-pack review that covers several sauces counts toward each. The "
               "one-sauce view drops those reviews; it is the cleaner comparison and the one "
               "the overview shows.", className="body"),
        dcc.RadioItems(
            id="heat-view",
            options=[{"label": "Reviews about one sauce", "value": "single"},
                     {"label": "All heat reviews", "value": "all"}],
            value="single", className="pills", inputClassName="pill-input",
            labelClassName="pill",
        ),
        graph(heat_gap_figure(data.heat_single), id="heat-gap"),
        html.P("The capped horizontal lines are 95% Wilson confidence intervals (CI) for "
               "the estimated share of heat reviews calling each sauce milder than expected. "
               "They show uncertainty in that proportion, not outliers or the spread of "
               "individual ratings.", className="note"),
        html.Div(id="heat-stats", children=heat_stats(data, "single")),
        heat_detail_table(data),
        html.P(f"Bars are hidden below {MIN_N_BAR} heat reviews; the table keeps every "
               "cell with at least five.", className="note"),
    ])


def split_stats(cp):
    if cp.get("status") != "ok":
        return []
    b, a = cp["before"], cp["after"]
    stats = [
        ("Before", f"{b['rate']:.0%} milder than expected", f"n = {b['n']}"),
        ("After", f"{a['rate']:.0%} milder than expected", f"n = {a['n']}"),
        ("Average stars", f"{b['mean_rating']:.2f} \u2192 {a['mean_rating']:.2f}",
         "before \u2192 after"),
    ]
    return [html.Div(className="hero-stats", children=[
        html.Div(className="hero-stat", children=[
            html.Div(k, className="muted small"),
            html.Div(v, className="mono"),
            html.Div(n, className="muted small"),
        ]) for k, v, n in stats
    ])]


def tech_time(data):
    cp = data.changepoint
    if cp.get("status") == "ok":
        b, a = cp["before"], cp["after"]
        verdict = (
            f"Best-fitting split {cp['change_date']} (bootstrap 95% interval "
            f"{cp['change_date_ci'][0]} to {cp['change_date_ci'][1]}); before "
            f"{b['rate']:.1%} (n={b['n']}), after {a['rate']:.1%} (n={a['n']}); "
            f"likelihood ratio {cp['lr_statistic']:.2f}, permutation {fmt_p(cp['p_value'])} "
            f"over {cp['n_permutations']:,} shuffles."
        )
        reading = ("No evidence of a change: the gap between Super Hot's name and how hot "
                   "it felt is persistent across the whole period."
                   if cp["p_value"] >= 0.05 else
                   "The share of milder-than-expected reviews shifted around this date.")
    else:
        verdict, reading = "Pending.", ""
    return tech_section("tech-time", "Change over time", [
        html.P("Did Super Hot read as milder after a recipe change? Among Super Hot heat "
               "reviews, a single change point in the milder-than-expected rate is located "
               "by maximum likelihood (at least 15 reviews per side), tested against "
               "shuffled dates, and dated by bootstrap.", className="body"),
        graph(changepoint_figure(cp)),
        *split_stats(cp),
        html.P(verdict, className="mono small muted"),
        html.P(reading, className="callout") if reading else None,
        html.P("Line: trailing six-month share, with Wilson 95% band; months with fewer than "
               "five heat reviews in the window are not plotted.", className="note"),
    ])


def tech_validation(data):
    v = data.validation
    kap, agr = v.get("kappa", {}), v.get("agreement", {})
    fields = [("heat", "Heat"), ("flavor", "Flavor"), ("vinegar", "Vinegar"),
              ("versatility", "Versatility"), ("value", "Value"),
              ("heat_direction", "Heat direction"),
              ("mentions_offspring", "Mentions The Offspring"),
              ("mentions_reorder", "Mentions reordering"),
              ("mentions_recipe_change", "Mentions a recipe change")]
    k_rows = [[label, num(kap.get(f), "{:.2f}"), kappa_band(kap.get(f)),
               pct(agr.get(f), 1)] for f, label in fields if f in kap]
    rho, rn = v.get("star_rho", {}), v.get("star_n", {})
    r_rows = [[ASPECT_LABELS[a], num(rho.get(a), "{:.2f}"), num(rn.get(a))]
              for a in ASPECT_LABELS if a in rho]
    return tech_section("tech-validation", "Label validation", [
        html.P(f"No reviews were hand coded. Two independent checks instead: a second model "
               f"({model_name(v.get('check_model')) or 'pending'}) relabeled a random "
               f"{num(v.get('n_overlap'))} reviews with the same prompt and schema, and each "
               "aspect's sentiment was compared with the reviewer's star rating.",
               className="body"),
        html.H3("Agreement between the two models", className="h3"),
        tech_table(["Field", "Cohen's kappa", "Landis & Koch", "Raw agreement"], k_rows,
                   numeric=("Cohen's kappa", "Raw agreement")) if k_rows else
        html.P("Pending.", className="muted small"),
        html.P("Rare fields can show a low kappa despite near-perfect raw agreement: with "
               "almost no positives in the sample, one disagreement dominates. Recipe-change "
               "mentions are reported as counts only for that reason.", className="note"),
        html.H3("Sentiment against star ratings", className="h3"),
        tech_table(["Aspect", "Spearman rho", "Reviews"], r_rows,
                   numeric=("Spearman rho", "Reviews")) if r_rows else
        html.P("Pending.", className="muted small"),
        html.P("Among reviews that mention the aspect; positive = +1, mixed = 0, negative = "
               "−1.", className="note"),
    ])


def tech_aspects(data):
    a = data.aspects
    if a.empty:
        return tech_section("tech-aspects", "Aspect detail", [html.P("Pending.")])
    rows = [[SAUCE_NAMES[r["sauce"]], ASPECT_LABELS[r["aspect"]], num(r["n_mentions"]),
             pct(r["mention_share"]), num(r["net"], "{:+.2f}"), pct(r["neg_share"])]
            for _, r in a.iterrows()]
    return tech_section("tech-aspects", "Aspect detail", [
        html.P("Net sentiment is the mean of +1 (positive), 0 (mixed) and −1 "
               "(negative) over reviews that mention the aspect. A review counts toward "
               "every sauce it discusses.", className="body"),
        tech_table(["Sauce", "Aspect", "Mentions", "Share of reviews", "Net", "Negative"],
                   rows, numeric=("Mentions", "Share of reviews", "Net", "Negative")),
    ])


def tech_scoville(data):
    h = data.heat
    src = data.summary.get("shu_sources", {})
    rows = ([[SAUCE_NAMES[r["sauce"]], num(r["labeled_shu"]), num(r["earlier_shu"]),
              num(r["lab_shu"]) if _ok(r.get("lab_shu")) else "pending"]
             for _, r in h.iterrows()] if not h.empty else [])
    return tech_section("tech-scoville", "Scoville values", [
        html.P("Label values have risen since the brand's earlier published tests. An HPLC "
               "capsaicinoid test of bottles bought now would add a third, independent "
               "measure.", className="body"),
        tech_table(["Sauce", "Label today (SHU)", "Earlier test", "Lab HPLC"], rows,
                   numeric=("Label today (SHU)", "Earlier test", "Lab HPLC"))
        if rows else html.P("Pending.", className="muted small"),
        html.P(f"Label: {src.get('labeled', 'brand store')}. Earlier: "
               f"{src.get('earlier', 'Pepper Geek')}.", className="note"),
    ])


def tech_data(data):
    return tech_section("tech-data", "Data notes", [
        html.P(["Source: ", data.summary.get("citation", "McAuley Lab, Amazon Reviews’23"),
                " Academic, non-commercial release; this page shows aggregates only and no "
                "review text."], className="body"),
        html.P(f"Limits: reviews end in {years(data)[1] or 'the release year'} and are "
               "self-selected; the labels are model "
               "output checked as above, not hand coded; Spicy Yellow is newer and has few "
               "reviews; a review's heat verdict applies to the whole review.",
               className="body"),
    ])


def technical_layout(app, data, home, tech_path):
    return html.Div(className="page-body", children=[
        tech_header(app),
        tech_nav(home),
        tech_heat(data),
        tech_time(data),
        tech_validation(data),
        tech_aspects(data),
        tech_scoville(data),
        tech_methods(data),
        tech_data(data),
        footer(data, tech_path),
    ])


################################################################################
# App
################################################################################
def build_app(data_dir=None, prefix: str | None = None) -> Dash:
    prefix = prefix or PREFIX
    data = load_data(data_dir)
    home, tech_path = prefix, f"{prefix}technical"

    ## suppress_callback_exceptions: the two pages render different components,
    ## so each page's callbacks target ids the other lacks.
    app = Dash(__name__, requests_pathname_prefix=prefix, title=TITLE,
               assets_folder=str(HERE / "assets"), external_stylesheets=[FONTS],
               suppress_callback_exceptions=True)
    app.index_string = app.index_string.replace(
        "{%favicon%}",
        f'<link rel="icon" type="image/svg+xml" '
        f'href="{app.get_asset_url("gb_reviews_icon.svg")}">',
    )
    app.layout = html.Div(className="wrap", children=[
        dcc.Location(id="url", refresh=False), dcc.Store(id="scroll-done"),
        html.Div(id="page"),
    ])

    @app.callback(Output("page", "children"), Input("url", "pathname"))
    def render_page(pathname):
        if pathname and pathname.rstrip("/").endswith("technical"):
            return technical_layout(app, data, home, tech_path)
        return overview_layout(app, data, home, tech_path)

    ## After a page renders, scroll to the #anchor in the URL (drill-down links),
    ## or to the top when switching pages without one.
    app.clientside_callback(
        """
        function(children, hash) {
            setTimeout(function() {
                if (hash) {
                    var el = document.getElementById(hash.slice(1));
                    if (el) { el.scrollIntoView(); return; }
                }
                window.scrollTo(0, 0);
            }, 350);
            return null;
        }
        """,
        Output("scroll-done", "data"),
        Input("page", "children"),
        State("url", "hash"),
    )

    @app.callback(
        Output("foods", "children"),
        Output("rivals", "children"),
        Output("fans", "children"),
        Input("sauce-filter", "value"),
    )
    def filter_scope(scope):
        return foods_children(data, scope), rivals_children(data, scope), fans_children(data, scope)

    @app.callback(
        Output("heat-gap", "figure"),
        Output("heat-stats", "children"),
        Input("heat-view", "value"),
    )
    def switch_heat_view(view):
        t = data.heat_single if view == "single" else data.heat
        return heat_gap_figure(t), heat_stats(data, view)

    @app.callback(Output("dl-report-file", "data"), Input("dl-report", "n_clicks"),
                  prevent_initial_call=True)
    def download_report(_n):
        if report_card is None:
            return no_update
        from flask import request
        scheme = request.headers.get("X-Forwarded-Proto", request.scheme)
        url = f"{scheme}://{request.host}{prefix}"
        pdf = report_card.build_report_card(data, url=url, author=AUTHOR)
        return dcc.send_bytes(pdf, "gringo_bandito_customer_report_card.pdf")

    return app


app = build_app()
server = app.server


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)
