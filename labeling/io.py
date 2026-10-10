"""Load label files into a flat DataFrame, one row per review."""

import json
from pathlib import Path

import pandas as pd

from core.constants import ASPECTS

SAUCE_KEYS = ("original", "green", "super_hot", "spicy_yellow")


def load_labels(path: Path) -> pd.DataFrame:
    """Latest successful label per review, flattened into columns."""
    latest = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("labels") is not None:
                latest[rec["review_id"]] = rec

    rows = []
    for rid, rec in latest.items():
        lab = rec["labels"]
        aspects = lab.get("aspects", {})
        row = {"review_id": rid, "label_model": rec.get("model")}
        for a in ASPECTS:
            row[a] = aspects.get(a, "not_mentioned")
        row["heat_direction"] = lab.get("heat_direction", "not_mentioned")
        row["sauces_discussed"] = list(lab.get("sauces_discussed") or [])
        row["peppers_mentioned"] = [p.lower() for p in lab.get("peppers_mentioned") or []]
        row["foods"] = [x.lower().strip() for x in lab.get("foods") or []]
        row["competitors"] = list(lab.get("competitors") or [])
        row["mentions_recipe_change"] = bool(lab.get("mentions_recipe_change"))
        row["mentions_offspring"] = bool(lab.get("mentions_offspring"))
        row["mentions_reorder"] = bool(lab.get("mentions_reorder"))
        rows.append(row)
    return pd.DataFrame(rows)


def attributed_sauces(listing_sauce: str, discussed: list) -> list:
    """Sauces a review counts toward: what it discusses, else its listing."""
    named = [s for s in discussed if s in SAUCE_KEYS]
    if named:
        return named
    return [listing_sauce] if listing_sauce in SAUCE_KEYS else []


def join_reviews_labels(reviews: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Inner join, adding the list of sauces each review is attributed to."""
    df = reviews.merge(labels, on="review_id", how="inner")
    df["sauces"] = [
        attributed_sauces(s, d) for s, d in zip(df["sauce"], df["sauces_discussed"])
    ]
    return df
