#!/usr/bin/env python3
"""
validate.py
-----------
Two checks on the automated labels, with no hand coding:

1. Inter-model agreement: Cohen's kappa between the primary and check models
   on the reviews both labeled, per aspect and for heat direction.
2. Convergent validity: aspect sentiment should rise with star rating
   (Spearman rho among reviews that mention the aspect).

Example usage:
    python modeling/validate.py \
        --reviews-file ./data/processed/reviews.parquet \
        --primary-labels ./data/interim/labels_primary.jsonl \
        --check-labels ./data/interim/labels_check.jsonl \
        --output-file ./reports/eval/validation.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from sklearn.metrics import cohen_kappa_score

from core.constants import ASPECTS
from labeling.io import load_labels
from modeling.stats import SENTIMENT_SCORE, spearman

app = typer.Typer(add_completion=False)

BOOL_FIELDS = ("mentions_recipe_change", "mentions_offspring", "mentions_reorder")


def kappa(a: pd.Series, b: pd.Series) -> dict:
    """Cohen's kappa plus raw agreement; NaN when only one label occurs."""
    agree = float((a == b).mean()) if len(a) else float("nan")
    if len(a) < 2 or len(set(a) | set(b)) < 2:
        return {"kappa": float("nan"), "agreement": agree, "n": int(len(a))}
    return {
        "kappa": float(cohen_kappa_score(a, b)),
        "agreement": agree,
        "n": int(len(a)),
    }


def agreement_report(primary: pd.DataFrame, check: pd.DataFrame) -> dict:
    both = primary.merge(check, on="review_id", suffixes=("_p", "_c"))
    out = {"n_overlap": int(len(both)), "fields": {}}
    for f in list(ASPECTS) + ["heat_direction"] + list(BOOL_FIELDS):
        out["fields"][f] = kappa(both[f"{f}_p"].astype(str), both[f"{f}_c"].astype(str))
    # Mention-level agreement: does the aspect come up at all?
    for a in ASPECTS:
        out["fields"][f"{a}_mentioned"] = kappa(
            both[f"{a}_p"] != "not_mentioned", both[f"{a}_c"] != "not_mentioned"
        )
    return out


def star_report(reviews: pd.DataFrame, labels: pd.DataFrame) -> dict:
    df = reviews[["review_id", "rating"]].merge(labels, on="review_id")
    out = {}
    for a in ASPECTS:
        m = df[df[a].isin(SENTIMENT_SCORE)]
        res = spearman(m[a].map(SENTIMENT_SCORE), m["rating"])
        res["mean_rating_by_sentiment"] = {
            s: float(m.loc[m[a] == s, "rating"].mean()) if (m[a] == s).any() else None
            for s in SENTIMENT_SCORE
        }
        out[a] = res
    return out


def kappa_band(k: float) -> str:
    """Landis and Koch descriptors."""
    if np.isnan(k):
        return "n/a"
    for cut, name in ((0.8, "almost perfect"), (0.6, "substantial"),
                      (0.4, "moderate"), (0.2, "fair"), (0.0, "slight")):
        if k > cut:
            return name
    return "poor"


@app.command()
def main(
    reviews_file: Path = typer.Option(Path("data/processed/reviews.parquet"), "--reviews-file"),
    primary_labels: Path = typer.Option(Path("data/interim/labels_primary.jsonl"), "--primary-labels"),
    check_labels: Path = typer.Option(Path("data/interim/labels_check.jsonl"), "--check-labels"),
    output_file: Path = typer.Option(Path("reports/eval/validation.json"), "--output-file"),
) -> None:
    """Report label agreement and star-rating validity."""
    reviews = pd.read_parquet(reviews_file)
    primary = load_labels(primary_labels)
    report = {
        "n_reviews": int(len(reviews)),
        "n_labeled_primary": int(len(primary)),
        "primary_model": primary["label_model"].mode().iat[0] if len(primary) else None,
        "label_coverage": float(len(primary) / max(1, len(reviews))),
        "star_validity": star_report(reviews, primary),
    }
    if check_labels.is_file():
        check = load_labels(check_labels)
        report["check_model"] = check["label_model"].mode().iat[0] if len(check) else None
        report["agreement"] = agreement_report(primary, check)
    else:
        typer.echo(f"No check labels at {check_labels}; skipping agreement.")

    typer.echo(f"Labeled {report['n_labeled_primary']:,} of {report['n_reviews']:,} reviews")
    if "agreement" in report:
        typer.echo(f"\nInter-model agreement on {report['agreement']['n_overlap']:,} reviews:")
        for f, r in report["agreement"]["fields"].items():
            typer.echo(
                f"  {f:28} kappa {r['kappa']:.2f} ({kappa_band(r['kappa'])}), "
                f"agreement {r['agreement']:.1%}"
            )
    typer.echo("\nAspect sentiment vs stars (Spearman):")
    for a, r in report["star_validity"].items():
        typer.echo(f"  {a:12} rho {r['rho']:.2f}  n={r['n']}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, indent=2, default=float))
    typer.echo(f"\nSaved {output_file}")


if __name__ == "__main__":
    app()
