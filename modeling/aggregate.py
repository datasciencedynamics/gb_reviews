#!/usr/bin/env python3
"""
aggregate.py
------------
Turns reviews + labels + model outputs into the small files the Dash app
reads. Every table carries its counts, so each number on the page can be
traced to how many reviews produced it. Review text never leaves this step.

Scopes: "all" plus one per sauce. A review counts toward each sauce it is
attributed to (see labeling.io.attributed_sauces).

Example usage:
    python modeling/aggregate.py \
        --reviews-file ./data/processed/reviews.parquet \
        --labels-file ./data/interim/labels_primary.jsonl \
        --validation-file ./reports/eval/validation.json \
        --changepoint-file ./reports/eval/changepoint_super_hot.json \
        --output-dir ./data/app
"""

import json
import unicodedata
from pathlib import Path

import pandas as pd
import typer

from core.config import DATASET_CITATION
from core.constants import (
    ASPECTS,
    COMPETITORS,
    EARLIER_SHU,
    LAB_SHU,
    LABELED_SHU,
    SAUCE_LABELS,
    SHU_SOURCES,
)
from labeling.io import SAUCE_KEYS, join_reviews_labels, load_labels
from modeling.stats import net_sentiment, wilson_ci

app = typer.Typer(add_completion=False)

SCOPES = ("all",) + SAUCE_KEYS
MIN_N = 5  # suppress cells built on fewer reviews


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return "".join(ch for ch in s.lower() if ch.isalnum())


_CANON = {_fold(c): c for c in COMPETITORS}


def canonical_brand(name: str) -> str:
    """Map spelling variants (tapatio, Tapatío) onto one display name."""
    f = _fold(name)
    for key, canon in _CANON.items():
        if key and (key in f or f in key) and len(f) >= 4:
            return canon
    return name.strip().title()


def scoped(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "all":
        return df
    return df[df["sauces"].map(lambda s: scope in s)]


def summary_table(df: pd.DataFrame) -> dict:
    out = {}
    for scope in SCOPES:
        d = scoped(df, scope)
        out[scope] = {
            "n": int(len(d)),
            "avg_rating": float(d["rating"].mean()) if len(d) else None,
            "verified_share": float(d["verified_purchase"].mean()) if len(d) else None,
            "heat_mention_share": float((d["heat"] != "not_mentioned").mean()) if len(d) else None,
            "date_min": str(d["date"].min().date()) if len(d) else None,
            "date_max": str(d["date"].max().date()) if len(d) else None,
        }
    return out


def aspect_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in SAUCE_KEYS:
        d = scoped(df, scope)
        for a in ASPECTS:
            net, n_m = net_sentiment(d[a])
            k_neg = int((d[a] == "negative").sum())
            rows.append({
                "sauce": scope,
                "aspect": a,
                "n_reviews": int(len(d)),
                "n_mentions": n_m,
                "mention_share": n_m / len(d) if len(d) else float("nan"),
                "net": net if n_m >= MIN_N else float("nan"),
                "neg_share": k_neg / n_m if n_m >= MIN_N else float("nan"),
            })
    return pd.DataFrame(rows)


def heat_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in SAUCE_KEYS:
        d = scoped(df, s)
        h = d[d["heat_direction"] != "not_mentioned"]
        n = len(h)
        row = {
            "sauce": s,
            "label": SAUCE_LABELS[s],
            "labeled_shu": LABELED_SHU[s],
            "earlier_shu": EARLIER_SHU[s],
            "lab_shu": LAB_SHU[s],
            "n_heat": n,
        }
        for direction in ("too_mild", "as_expected", "too_hot"):
            k = int((h["heat_direction"] == direction).sum())
            lo, hi = wilson_ci(k, n)
            ok = n >= MIN_N
            row[direction] = k / n if ok else float("nan")
            row[f"{direction}_lo"] = lo if ok else float("nan")
            row[f"{direction}_hi"] = hi if ok else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def foods_table(df: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    rows = []
    for scope in SCOPES:
        d = scoped(df, scope)
        counts = d["foods"].explode().dropna().value_counts()
        counts = counts[counts >= 2].head(top)
        rows += [{"scope": scope, "food": f, "n": int(n)} for f, n in counts.items()]
    return pd.DataFrame(rows, columns=["scope", "food", "n"])


def competitor_table(df: pd.DataFrame, top: int = 8) -> pd.DataFrame:
    rows = []
    for scope in SCOPES:
        d = scoped(df, scope)
        comps = [
            (canonical_brand(c["brand"]), c["comparison"])
            for cs in d["competitors"]
            for c in cs
            if c.get("brand")
        ]
        if not comps:
            continue
        c = pd.DataFrame(comps, columns=["brand", "comparison"])
        g = c.groupby("brand")["comparison"].value_counts().unstack(fill_value=0)
        for col in ("gb_better", "gb_worse", "similar", "unclear"):
            if col not in g:
                g[col] = 0
        g["n"] = g[["gb_better", "gb_worse", "similar", "unclear"]].sum(axis=1)
        g = g.sort_values("n", ascending=False).head(top)
        for brand, r in g.iterrows():
            rows.append({
                "scope": scope,
                "brand": brand,
                "n": int(r["n"]),
                "gb_better": int(r["gb_better"]),
                "gb_worse": int(r["gb_worse"]),
                "similar": int(r["similar"]),
                "unclear": int(r["unclear"]),
            })
    return pd.DataFrame(
        rows,
        columns=["scope", "brand", "n", "gb_better", "gb_worse", "similar", "unclear"],
    )


def fans_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in SCOPES:
        d = scoped(df, scope)
        for group, sub in (("Mentions The Offspring", d[d["mentions_offspring"]]),
                           ("Everyone else", d[~d["mentions_offspring"]])):
            top = sub["sauces"].explode().dropna().value_counts()
            rows.append({
                "scope": scope,
                "group": group,
                "n": int(len(sub)),
                "avg_rating": float(sub["rating"].mean()) if len(sub) >= MIN_N else float("nan"),
                "reorder_share": float(sub["mentions_reorder"].mean()) if len(sub) >= MIN_N else float("nan"),
                "top_sauce": SAUCE_LABELS[top.index[0]] if scope == "all" and len(top) else "",
            })
    return pd.DataFrame(rows)


def validation_summary(path: Path) -> dict:
    if not path.is_file():
        return {}
    v = json.loads(path.read_text())
    out = {
        "primary_model": v.get("primary_model"),
        "check_model": v.get("check_model"),
        "label_coverage": v.get("label_coverage"),
        "star_rho": {a: r["rho"] for a, r in v.get("star_validity", {}).items()},
    }
    agr = v.get("agreement")
    if agr:
        out["n_overlap"] = agr["n_overlap"]
        out["kappa"] = {f: r["kappa"] for f, r in agr["fields"].items()}
    return out


@app.command()
def main(
    reviews_file: Path = typer.Option(Path("data/processed/reviews.parquet"), "--reviews-file"),
    labels_file: Path = typer.Option(Path("data/interim/labels_primary.jsonl"), "--labels-file"),
    validation_file: Path = typer.Option(Path("reports/eval/validation.json"), "--validation-file"),
    changepoint_file: Path = typer.Option(Path("reports/eval/changepoint_super_hot.json"), "--changepoint-file"),
    output_dir: Path = typer.Option(Path("data/app"), "--output-dir"),
) -> None:
    """Write the app's data artifacts."""
    df = join_reviews_labels(pd.read_parquet(reviews_file), load_labels(labels_file))
    output_dir.mkdir(parents=True, exist_ok=True)

    aspect_table(df).to_parquet(output_dir / "aspects.parquet", index=False)
    heat_table(df).to_parquet(output_dir / "heat.parquet", index=False)
    foods_table(df).to_parquet(output_dir / "foods.parquet", index=False)
    competitor_table(df).to_parquet(output_dir / "competitors.parquet", index=False)
    fans_table(df).to_parquet(output_dir / "fans.parquet", index=False)

    summary = {
        "scopes": summary_table(df),
        "citation": DATASET_CITATION,
        "shu_sources": SHU_SOURCES,
        "validation": validation_summary(validation_file),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    if changepoint_file.is_file():
        (output_dir / "changepoint.json").write_text(changepoint_file.read_text())
    else:
        typer.echo(f"No change-point file at {changepoint_file}; the app will show a placeholder.")

    typer.echo(f"Wrote app artifacts for {len(df):,} labeled reviews to {output_dir}")


if __name__ == "__main__":
    app()
