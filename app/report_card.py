"""
report_card.py
--------------
One-page "Customer report card" PDF for the Gringo Bandito Review Lab.

Built from the same data the app shows (the namespace app.load_data returns),
so it updates whenever the pipeline reruns. Same identity as the app: dark
masthead with the chili mark, Anton headlines, IBM Plex for text and numbers,
on a light page that prints and forwards well.

Fonts and images live in assets/ next to this file (fonts/ holds the TTFs and
their OFL licenses). Missing fonts fall back to Helvetica and Courier.

    from report_card import build_report_card
    pdf_bytes = build_report_card(data, url="https://...", author="...")
"""

import io
import math
from pathlib import Path

import pandas as pd
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ASSETS = Path(__file__).resolve().parent / "assets"

################################################################################
# Fonts
################################################################################
_FONT_FILES = {
    "Display": "Anton-Regular.ttf",
    "Sans": "IBMPlexSans-Regular.ttf",
    "SansBold": "IBMPlexSans-SemiBold.ttf",
    "Mono": "IBMPlexMono-Regular.ttf",
    "MonoBold": "IBMPlexMono-Medium.ttf",
}
_FALLBACK = {"Display": "Helvetica-Bold", "Sans": "Helvetica", "SansBold": "Helvetica-Bold",
             "Mono": "Courier", "MonoBold": "Courier-Bold"}
F = {}
for _key, _file in _FONT_FILES.items():
    _path = ASSETS / "fonts" / _file
    try:
        _name = f"GB-{_key}"
        if _name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(_name, str(_path)))
        F[_key] = _name
    except Exception:  # noqa: BLE001 - any font problem falls back cleanly
        F[_key] = _FALLBACK[_key]

################################################################################
# Look
################################################################################
NIGHT = HexColor("#16120F")
CREAM = HexColor("#F2EDE4")
PAPER = HexColor("#FBF8F3")
INK = HexColor("#1C1A17")
INK2 = HexColor("#4A453E")
MUTED = HexColor("#8A8277")
RULE = HexColor("#DDD5C8")
TRACK = HexColor("#ECE6DC")
ORANGE = HexColor("#E8701F")
ORANGE_DARK = HexColor("#C2541A")
GOLD = HexColor("#F0B64A")
BLUE = HexColor("#4A86C5")

SAUCES = ("original", "green", "super_hot", "spicy_yellow")
SAUCE_NAMES = {"original": "Original Red", "green": "Green Habanero",
               "super_hot": "Super Hot", "spicy_yellow": "Spicy Yellow"}
SAUCE_COLORS = {"original": "#D7392B", "green": "#4E9A55",
                "super_hot": "#EE7A2B", "spicy_yellow": "#E3B23C"}
ASPECTS = (("flavor", "Flavor"), ("heat", "Heat"), ("vinegar", "Vinegar"),
           ("versatility", "Versatility"), ("value", "Value"))

W, H = letter
M = 44  # page margin


def _ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def _model_name(model_id):
    """'claude-haiku-5-5' -> 'Claude Haiku 5.5'."""
    if not model_id:
        return None
    parts = model_id.split("-")
    if parts[0] == "claude" and len(parts) >= 4:
        return f"Claude {parts[1].title()} {'.'.join(parts[2:])}"
    return model_id


################################################################################
# Drawing helpers
################################################################################
def _text(c, x, y, s, size, color=INK, font="Sans", align="left", tracking=0):
    c.setFillColor(color)
    if tracking:
        ## Character spacing lives in the PDF graphics state and would leak into
        ## every later string, so it is scoped with save/restore.
        c.saveState()
        t = c.beginText()
        t.setFont(F[font], size)
        t.setCharSpace(tracking)
        width = c.stringWidth(s, F[font], size) + tracking * (len(s) - 1)
        x = x - width if align == "right" else x - width / 2 if align == "center" else x
        t.setTextOrigin(x, y)
        t.textOut(s)
        c.drawText(t)
        c.restoreState()
        c.setFillColor(color)
        return
    c.setFont(F[font], size)
    if align == "right":
        c.drawRightString(x, y, s)
    elif align == "center":
        c.drawCentredString(x, y, s)
    else:
        c.drawString(x, y, s)


def _wrap(c, x, y, s, width, size, color=INK2, font="Sans", leading=None):
    """Draw wrapped text; returns the y below the last line."""
    leading = leading or size * 1.45
    c.setFont(F[font], size)
    c.setFillColor(color)
    line = ""
    for w in s.split():
        trial = f"{line} {w}".strip()
        if c.stringWidth(trial, F[font], size) <= width:
            line = trial
        else:
            c.drawString(x, y, line)
            y -= leading
            line = w
    if line:
        c.drawString(x, y, line)
        y -= leading
    return y


def _rule(c, x0, x1, y, color=RULE, width=0.7):
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.line(x0, y, x1, y)


def _heading(c, x, y, s, x_end=None):
    _text(c, x, y, s.upper(), 13, INK, "Display", tracking=0.6)
    _rule(c, x, x_end or W - M, y - 7, INK, 1.1)
    return y - 26


def _dot(c, x, y, sauce, r=3.6):
    c.setFillColor(HexColor(SAUCE_COLORS[sauce]))
    c.circle(x, y, r, stroke=0, fill=1)


def _bar(c, x, y, width, share, color, height=3.2):
    c.setFillColor(TRACK)
    c.rect(x, y, width, height, stroke=0, fill=1)
    c.setFillColor(color)
    c.rect(x, y, max(1.5, width * max(0.0, min(1.0, share))), height, stroke=0, fill=1)


def _restore_text(node, letter_spacing, font):
    """svglib drops font-weight and letter-spacing on SVG text. Rebuild each
    string in the bold face, one glyph at a time, with the original spacing."""
    from reportlab.graphics.shapes import Group, String
    if not hasattr(node, "contents"):
        return
    new = []
    for child in node.contents:
        if isinstance(child, String):
            x = child.x
            g = Group()
            for ch in child.text:
                g.add(String(x, child.y, ch, fontName=font, fontSize=child.fontSize,
                             fillColor=child.fillColor))
                x += pdfmetrics.stringWidth(ch, font, child.fontSize) + letter_spacing
            new.append(g)
        else:
            _restore_text(child, letter_spacing, font)
            new.append(child)
    node.contents = new


def _svg(c, name, x, y, w, h, letter_spacing=0, font=None):
    """Draw an SVG from assets/ as vector art, fitted into the w x h box.
    Needs svglib; without it the box falls back to a plain text mark."""
    drawing = None
    try:
        from reportlab.graphics import renderPDF
        from svglib.svglib import svg2rlg
        drawing = svg2rlg(str(ASSETS / name))
    except Exception:  # noqa: BLE001 - missing svglib or file falls back to text
        pass
    if drawing is None:
        _text(c, x + w, y + h / 2 - 4, "DATA SCIENCE DYNAMICS", 10, HexColor("#185FA5"),
              "MonoBold", "right")
        return
    if font:
        _restore_text(drawing, letter_spacing, font)
    k = min(w / drawing.width, h / drawing.height)
    dw, dh = drawing.width * k, drawing.height * k
    drawing.width, drawing.height = dw, dh
    drawing.scale(k, k)
    renderPDF.draw(drawing, c, x + (w - dw), y + (h - dh) / 2)


def _image(c, name, x, y, w, h):
    path = ASSETS / name
    if path.is_file():
        c.drawImage(str(path), x, y, w, h, mask="auto", preserveAspectRatio=True, anchor="c")


################################################################################
# Report card
################################################################################
def build_report_card(data, url: str = "", author: str = "") -> bytes:
    s = data.summary.get("scopes", {}).get("all", {})
    y0 = (s.get("date_min") or "")[:4]
    y1 = (s.get("date_max") or "")[:4]
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.setTitle("Gringo Bandito customer report card")
    c.setAuthor(author or "Data Science Dynamics")

    c.setFillColor(PAPER)
    c.rect(0, 0, W, H, stroke=0, fill=1)

    # ── Masthead ───────────────────────────────────────────────────────────
    mh = 104
    c.setFillColor(ORANGE)
    c.rect(0, H - mh - 3, W, 3, stroke=0, fill=1)
    tx = M
    _text(c, tx, H - 38, "INDEPENDENT REVIEW ANALYSIS  /  GRINGO BANDITO HOT SAUCE", 8,
          ORANGE_DARK, "MonoBold", tracking=1.4)
    _text(c, tx, H - 70, "CUSTOMER REPORT CARD", 32, INK, "Display", tracking=0.8)
    bits = []
    if s.get("n"):
        bits.append(f"{s['n']:,} Amazon reviews")
    if y0:
        bits.append(f"{y0} to {y1}")
    if _ok(s.get("avg_rating")):
        bits.append(f"{s['avg_rating']:.2f} stars on average")
    _text(c, tx, H - 88, "  /  ".join(bits), 9, MUTED, "Mono")
    ## letter-spacing 5 and bold Helvetica, as set in dsd_logo.svg
    _svg(c, "dsd_logo.svg", W - M - 160, H - mh + 30, 160, 44,
         letter_spacing=5, font="Helvetica-Bold")

    y = H - mh - 36

    # ── The big one ────────────────────────────────────────────────────────
    g = data.summary.get("heat_gap", {}).get("single", {})
    st = data.summary.get("super_hot_stars", {})
    _text(c, M, y, "THE ONE THING FANS ASK FOR", 8.5, MUTED, "MonoBold", tracking=1.4)
    y -= 30
    _text(c, M, y, "SUPER HOT FANS WANT IT HOTTER", 30, ORANGE, "Display", tracking=0.4)
    y -= 16
    stats = []
    if g:
        stats.append((f"{g['rate_a']:.0%}", "of Super Hot buyers",
                      "wanted more heat"))
        stats.append((f"{g['rate_b']:.0%}", "said the same about",
                      "Original Red"))
    mild, exp = st.get("too_mild"), st.get("as_expected")
    if mild and exp:
        stats.append((f"{mild['mean']:.1f}", "stars from buyers who",
                      "wanted more heat"))
        stats.append((f"{exp['mean']:.1f}", "stars when the heat",
                      "matched the name"))
    if stats:
        cw = (W - 2 * M) / len(stats)
        top = y
        for i, (big, l1, l2) in enumerate(stats):
            x = M + cw * i
            if i:
                c.setStrokeColor(RULE)
                c.setLineWidth(0.7)
                c.line(x, top - 2, x, top - 66)
            pad = 0 if i == 0 else 14
            color = ORANGE if i in (0, 3) else INK
            _text(c, x + pad, top - 34, big, 30, color, "MonoBold")
            _text(c, x + pad, top - 50, l1, 8.5, INK2)
            _text(c, x + pad, top - 61, l2, 8.5, INK2)
        y = top - 88
    y = _wrap(c, M, y, "Buyers still like Super Hot. Those who got the heat its name promises "
              "rated it nearly a full star higher, so living up to the name is worth about a "
              "star.", W - 2 * M, 10.5, INK, "Sans")
    y -= 12

    # ── Scorecard ──────────────────────────────────────────────────────────
    y = _heading(c, M, y, "Scorecard: share of praise by topic")
    a = data.aspects
    name_w = 128
    col_w = (W - 2 * M - name_w) / len(ASPECTS)
    for i, (_, label) in enumerate(ASPECTS):
        _text(c, M + name_w + col_w * i, y, label.upper(), 7.5, MUTED, "MonoBold", tracking=1)
    y -= 8
    _rule(c, M, W - M, y)
    row_h = 26
    for sauce in SAUCES:
        sub = a[a["sauce"] == sauce].set_index("aspect") if not a.empty else pd.DataFrame()
        cy = y - row_h / 2 - 2
        _dot(c, M + 4, cy + 3, sauce)
        _text(c, M + 14, cy, SAUCE_NAMES[sauce], 10.5, INK, "SansBold")
        for i, (key, _) in enumerate(ASPECTS):
            x0 = M + name_w + col_w * i
            r = sub.loc[key] if key in sub.index else None
            if r is not None and _ok(r.get("net")):
                praise = float(r["net"] + r["neg_share"])
                color = ORANGE if praise >= 0.5 else BLUE
                _text(c, x0, cy + 2, f"{praise:.0%}", 12, INK, "MonoBold")
                _bar(c, x0, cy - 7, col_w - 22, praise, color)
            else:
                _text(c, x0, cy + 2, "too few", 9, MUTED, "Mono")
        y -= row_h
        _rule(c, M, W - M, y)
    y = _wrap(c, M, y - 12, "Of the reviews that bring up each topic, the share that praise "
              "it. Orange bars lean praise, blue lean complaints. Spicy Yellow is the newest "
              "sauce and has the fewest reviews.", W - 2 * M, 8.5, MUTED)
    y -= 18

    # ── Where it wins / How people use it ──────────────────────────────────
    gap = 30
    col = (W - 2 * M - gap) / 2
    left, right = M, M + col + gap
    _heading(c, left, y, "Where it wins", left + col)
    y_list = _heading(c, right, y, "What it goes on", W - M)

    yl = y_list
    comp = data.competitors
    comp = comp[comp["scope"] == "all"].head(4) if not comp.empty else comp
    if comp.empty:
        _text(c, left, yl, "No competitor comparisons yet.", 9.5, MUTED)
    for _, r in comp.iterrows():
        share = r["gb_better"] / r["n"] if r["n"] else 0
        _text(c, left, yl, r["brand"], 10, INK, "SansBold")
        _text(c, left + col, yl, f"{share:.0%}", 10.5, ORANGE, "MonoBold", "right")
        _text(c, left + col - 34, yl, f"of {int(r['n'])}", 8.5, MUTED, "Mono", "right")
        _bar(c, left, yl - 7, col, share, ORANGE, 2.4)
        yl -= 21
    yl = _wrap(c, left, yl - 2, "Share of head-to-head mentions in which the reviewer preferred "
               "Gringo Bandito.", col, 8.5, MUTED)

    yr = y_list
    foods = data.foods
    foods = foods[foods["scope"] == "all"].head(6) if not foods.empty else foods
    if foods.empty:
        _text(c, right, yr, "No food mentions yet.", 9.5, MUTED)
    for i, (_, r) in enumerate(foods.iterrows(), 1):
        _text(c, right, yr, f"{i:02d}", 9, MUTED, "Mono")
        _text(c, right + 22, yr, str(r["food"]).capitalize(), 10, INK)
        _text(c, W - M, yr, f"{int(r['n'])}", 10, INK2, "Mono", "right")
        c.setStrokeColor(RULE)
        c.setDash(1, 2)
        c.line(right + 26 + c.stringWidth(str(r["food"]).capitalize(), F["Sans"], 10), yr + 2,
               W - M - 26, yr + 2)
        c.setDash()
        yr -= 14.5
    _text(c, right, yr - 2, "Mentions across all reviews.", 8.5, MUTED)
    y = min(yl, yr - 16) - 10

    # ── The fans (one line, gold rule) ─────────────────────────────────────
    fans = data.fans
    fans = fans[fans["scope"] == "all"].set_index("group") if not fans.empty else fans
    if not fans.empty and "Mentions The Offspring" in fans.index:
        f, o = fans.loc["Mentions The Offspring"], fans.loc["Everyone else"]
        fy = min(y, 136)
        _rule(c, M, W - M, fy + 12, GOLD, 1.2)
        _text(c, M, fy, "CAME FOR THE BAND", 8, MUTED, "MonoBold", tracking=1.2)
        line = (f"{int(f['n']):,} reviewers mention The Offspring or Dexter Holland: "
                f"{f['avg_rating']:.2f} stars and {f['reorder_share']:.0%} bought again, "
                f"against {o['avg_rating']:.2f} and {o['reorder_share']:.0%} for everyone else.")
        _wrap(c, M + 118, fy, line, W - 2 * M - 118, 9, INK2)

    # ── Footer ─────────────────────────────────────────────────────────────
    _rule(c, M, W - M, 92)
    v = data.summary.get("validation", {})
    primary = _model_name(v.get("primary_model")) or "a language model"
    check = _model_name(v.get("check_model"))
    note = ("Source: public Amazon customer reviews, via McAuley Lab's Amazon Reviews'23 "
            f"research dataset (Hou et al., 2024). Each review was labeled by {primary}"
            + (f"; {check} relabeled a 10% sample to check agreement" if check else "")
            + ". An independent analysis, not affiliated with or endorsed by Gringo Bandito "
            "or Dexter Holland.")
    _wrap(c, M, 78, note, W - 2 * M, 7.5, MUTED, "Sans", leading=10)
    who = "DATA SCIENCE DYNAMICS" + (f"   {author}" if author else "")
    _text(c, M, 40, who, 8, INK2, "MonoBold", tracking=0.6)
    if url:
        label = f"Interactive version: {url}"
        _text(c, M, 27, label, 8.5, ORANGE, "Mono")
        c.linkURL(url, (M, 23, M + c.stringWidth(label, F["Mono"], 8.5), 35), relative=0)

    c.showPage()
    c.save()
    return buf.getvalue()
