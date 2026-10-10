"""Plotly figures, styled to match the page."""

import pandas as pd
import plotly.graph_objects as go

INK = "#F2EDE4"
MUTED = "#A69E92"
GRID = "#3A322B"
ACCENT = "#EE7A2B"
GOLD = "#F0B64A"


def _base(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=56, r=16, t=16, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", color=MUTED, size=13),
        hoverlabel=dict(bgcolor="#211B17", bordercolor=GRID, font_color=INK),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickformat=".0%", rangemode="tozero")
    return fig


def empty_figure(message: str, height: int = 320) -> go.Figure:
    fig = _base(go.Figure(), height)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.add_annotation(text=message, showarrow=False, font=dict(size=15, color="#6B6157"))
    return fig


SAUCE_ORDER = ("original", "green", "spicy_yellow", "super_hot")
SAUCE_NAMES = {
    "original": "Original Red",
    "green": "Green Habanero",
    "super_hot": "Super Hot",
    "spicy_yellow": "Spicy Yellow",
}
MIN_N_BAR = 20  # fewer heat reviews than this: no bar, just a note


def heat_gap_figure(heat: pd.DataFrame) -> go.Figure:
    """Share of heat reviews calling each sauce milder than expected, with
    Wilson 95% intervals. Super Hot in accent, the rest neutral."""
    if heat.empty:
        return empty_figure("data pending", height=300)
    t = heat.set_index("sauce").reindex(SAUCE_ORDER)
    ok = t["n_heat"].fillna(0) >= MIN_N_BAR

    # The value and n sit in the axis label, so nothing collides with the
    # error bars and the chart keeps its width on a phone.
    ticks, rates, lo, hi, colors = [], [], [], [], []
    for s in SAUCE_ORDER:
        r = t.loc[s]
        n = int(r["n_heat"]) if pd.notna(r["n_heat"]) else 0
        if ok[s]:
            ticks.append(f"<b>{SAUCE_NAMES[s]}</b><br>{r['too_mild']:.0%} · n={n}")
            rates.append(r["too_mild"])
            lo.append(r["too_mild"] - r["too_mild_lo"])
            hi.append(r["too_mild_hi"] - r["too_mild"])
        else:
            ticks.append(f"<b>{SAUCE_NAMES[s]}</b><br>too few (n={n})")
            rates.append(0)
            lo.append(0)
            hi.append(0)
        colors.append(ACCENT if s == "super_hot" else "#8C8378")

    upper = max((r + h for r, h in zip(rates, hi)), default=0.5)
    fig = go.Figure(go.Bar(
        y=ticks, x=rates, orientation="h",
        marker=dict(color=colors),
        error_x=dict(type="data", symmetric=False, array=hi, arrayminus=lo,
                     color=INK, thickness=1.5, width=6),
        hovertemplate="%{x:.0%} milder than expected<extra></extra>",
    ))
    fig = _base(fig, 320)
    fig.update_layout(margin=dict(l=8, r=16, t=8, b=48), bargap=0.35)
    fig.update_xaxes(tickformat=".0%", range=[0, min(1, upper * 1.1 + 0.02)],
                     gridcolor=GRID, showgrid=True, nticks=6,
                     title_text="Milder than expected",
                     title_font=dict(size=12))
    fig.update_yaxes(showgrid=False, automargin=True, ticksuffix="  ",
                     tickfont=dict(color=INK, size=13, family="IBM Plex Mono, monospace"))
    return fig


def changepoint_figure(cp: dict) -> go.Figure:
    """Trailing six-month share of 'milder than expected' heat reviews."""
    series = pd.DataFrame(cp.get("series") or [])
    if series.empty:
        return empty_figure("data pending")
    series["month"] = pd.to_datetime(series["month"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.concat([series["month"], series["month"][::-1]]),
        y=pd.concat([series["hi"], series["lo"][::-1]]),
        fill="toself", fillcolor="rgba(238,122,43,0.18)",
        line=dict(width=0), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=series["month"], y=series["rate"], mode="lines",
        line=dict(color=ACCENT, width=2.5),
        customdata=series["n"],
        hovertemplate="%{x|%b %Y}<br>%{y:.0%} milder than expected<br>%{customdata} heat reviews in window<extra></extra>",
    ))

    if cp.get("status") == "ok":
        lo, hi = cp.get("change_date_ci") or (None, None)
        if lo and hi:
            fig.add_vrect(x0=lo, x1=hi, fillcolor=GOLD, opacity=0.10, line_width=0)
        fig.add_vline(x=cp["change_date"], line=dict(color=GOLD, width=2, dash="dash"))
        fig.add_annotation(
            x=cp["change_date"], y=1, yref="paper", xanchor="left", yanchor="top",
            text=f"  Estimated change {pd.to_datetime(cp['change_date']):%b %Y}",
            showarrow=False, font=dict(color=GOLD, size=13),
        )
    fig = _base(fig, 340)
    fig.update_yaxes(title_text="Milder than expected", title_font=dict(size=12))
    return fig
