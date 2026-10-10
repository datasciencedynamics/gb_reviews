#!/usr/bin/env python3
"""
preprocessing.py
----------------
Joins reviews to product metadata, maps each listing to a sauce, cleans the
text, drops duplicates and writes one Parquet file with one row per review.

Example usage:
    python preprocessing/preprocessing.py \
        --meta-file ./data/raw/gb_meta.jsonl \
        --reviews-file ./data/raw/gb_reviews.jsonl \
        --output-data-file ./data/processed/reviews.parquet
"""

import hashlib
import html
import re
from pathlib import Path

import pandas as pd
import typer

from core.constants import SAUCE_PATTERNS, CHILD_ASIN_SAUCE

app = typer.Typer(add_completion=False)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_text(text) -> str:
    """Unescape HTML, strip tags and collapse whitespace."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = html.unescape(str(text))
    text = _TAG_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def map_sauce(title: str) -> str:
    """Map a product title to one sauce key, 'variety' or 'unknown'."""
    t = (title or "").lower()
    if any(w in t for w in ("variety", "gift", "crate", "sampler", "collection")):
        return "variety"
    hits = [
        sauce
        for sauce, patterns in SAUCE_PATTERNS.items()
        if any(p in t for p in patterns)
    ]
    # "Super Hot" titles also contain "hot sauce"; patterns are specific
    # enough that a single hit is unambiguous.
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        # Original's generic "red hot sauce" can co-occur with a named sauce
        named = [h for h in hits if h != "original"]
        return named[0] if len(named) == 1 else "variety"
    # Untitled flavor usually means the flagship Original
    if "hot sauce" in t:
        return "original"
    return "unknown"


def review_id(row: pd.Series) -> str:
    """Stable id from user, item, time and text (the dataset has none)."""
    key = f"{row['user_id']}|{row['parent_asin']}|{row['timestamp']}|{row['text'][:200]}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def build_reviews(meta: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """Join, map sauces, clean text and return the analysis table."""
    meta = meta[["parent_asin", "title"]].rename(columns={"title": "product_title"})
    meta = meta.drop_duplicates("parent_asin")
    meta["sauce"] = meta["product_title"].map(map_sauce)

    df = reviews.merge(meta, on="parent_asin", how="left")
    df["sauce"] = df["sauce"].fillna("unknown")
    df["sauce"] = df["asin"].map(CHILD_ASIN_SAUCE).fillna(df["sauce"])

    for col, default in (("title", ""), ("helpful_vote", 0), ("asin", None)):
        if col not in df:
            df[col] = default
    df["title"] = df["title"].map(clean_text)
    df["text"] = df["text"].map(clean_text)
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["rating"] = df["rating"].astype(float)
    df["verified_purchase"] = df["verified_purchase"].astype(bool)

    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(["user_id", "parent_asin", "text"])
    df["review_id"] = df.apply(review_id, axis=1)

    cols = [
        "review_id",
        "parent_asin",
        "asin",
        "product_title",
        "sauce",
        "user_id",
        "date",
        "month",
        "rating",
        "title",
        "text",
        "verified_purchase",
        "helpful_vote",
    ]
    return df[cols].sort_values("date").reset_index(drop=True)


@app.command()
def main(
    meta_file: Path = typer.Option(Path("data/raw/gb_meta.jsonl"), "--meta-file"),
    reviews_file: Path = typer.Option(
        Path("data/raw/gb_reviews.jsonl"), "--reviews-file"
    ),
    output_data_file: Path = typer.Option(
        Path("data/processed/reviews.parquet"), "--output-data-file"
    ),
) -> None:
    """Build the cleaned review table."""
    for p in (meta_file, reviews_file):
        if not p.is_file():
            typer.secho(f"Input file not found: {p}", fg="red", err=True)
            raise typer.Exit(code=1)

    meta = pd.read_json(meta_file, lines=True)
    reviews = pd.read_json(reviews_file, lines=True, convert_dates=False)
    typer.echo(f"Loaded {len(meta)} products, {len(reviews):,} reviews")

    df = build_reviews(meta, reviews)

    typer.echo("\nListings by sauce:")
    listing = (
        df.groupby(["sauce", "product_title"]).size().rename("n").reset_index()
    )
    typer.echo(listing.sort_values(["sauce", "n"], ascending=[True, False]).to_string(index=False))
    typer.echo("\nReviews by sauce:")
    typer.echo(df["sauce"].value_counts().to_string())
    typer.echo(
        f"\nDate range: {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}"
    )
    typer.echo(f"Verified purchases: {df['verified_purchase'].mean():.1%}")

    output_data_file = output_data_file.resolve()
    output_data_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_data_file, index=False)
    typer.echo(f"\nSaved {len(df):,} reviews to {output_data_file}")


if __name__ == "__main__":
    app()
