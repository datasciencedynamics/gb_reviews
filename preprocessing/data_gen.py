#!/usr/bin/env python3
"""
data_gen.py
-----------
Streams the McAuley Lab Amazon Reviews'23 Grocery_and_Gourmet_Food files line
by line (nothing multi-GB is written to disk), keeps only Gringo Bandito
products, and writes two small JSONL files.

Example usage:
    python preprocessing/data_gen.py \
        --meta-out ./data/raw/gb_meta.jsonl \
        --reviews-out ./data/raw/gb_reviews.jsonl
"""

import gzip
import io
import json
from pathlib import Path
from typing import Iterable, Iterator

import requests
import typer

from core.config import META_URL, REVIEW_URL
from core.constants import BRAND_KEYS

app = typer.Typer(add_completion=False)


def stream_jsonl_gz(url: str) -> Iterator[dict]:
    """Yield parsed JSON objects from a remote .jsonl.gz without saving it."""
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        r.raw.decode_content = False
        with gzip.GzipFile(fileobj=r.raw) as gz:
            for line in io.TextIOWrapper(gz, encoding="utf-8"):
                if line.strip():
                    yield json.loads(line)


def is_brand_item(item: dict, keys: Iterable[str] = BRAND_KEYS) -> bool:
    """True when any brand key appears in the store, title or Brand detail."""
    details = item.get("details") or {}
    blob = " ".join(
        [
            str(item.get("store") or ""),
            str(item.get("title") or ""),
            str(details.get("Brand", "")),
        ]
    ).lower()
    return any(k in blob for k in keys)


@app.command()
def main(
    meta_out: Path = typer.Option(
        Path("data/raw/gb_meta.jsonl"),
        "--meta-out",
        help="Where matched product metadata is written.",
    ),
    reviews_out: Path = typer.Option(
        Path("data/raw/gb_reviews.jsonl"),
        "--reviews-out",
        help="Where reviews for matched products are written.",
    ),
    meta_url: str = typer.Option(META_URL, "--meta-url"),
    review_url: str = typer.Option(REVIEW_URL, "--review-url"),
) -> None:
    """Pull Gringo Bandito products and their reviews from Amazon Reviews'23."""

    meta_out = meta_out.resolve()
    reviews_out = reviews_out.resolve()
    meta_out.parent.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Scanning product metadata: {meta_url}")
    asins = set()
    with open(meta_out, "w", encoding="utf-8") as f:
        for i, item in enumerate(stream_jsonl_gz(meta_url), 1):
            if is_brand_item(item):
                asins.add(item["parent_asin"])
                f.write(json.dumps(item) + "\n")
            if i % 100_000 == 0:
                typer.echo(f"  {i:,} items scanned, {len(asins)} matches")
    typer.echo(f"Matched {len(asins)} products -> {meta_out}")

    if not asins:
        typer.secho(
            "No matches. Check BRAND_KEYS against the 'store' field.",
            fg="red",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(f"Scanning reviews (about 14M rows): {review_url}")
    kept = 0
    with open(reviews_out, "w", encoding="utf-8") as f:
        for i, rev in enumerate(stream_jsonl_gz(review_url), 1):
            if rev.get("parent_asin") in asins:
                f.write(json.dumps(rev) + "\n")
                kept += 1
            if i % 1_000_000 == 0:
                typer.echo(f"  {i:,} reviews scanned, {kept} kept")
    typer.echo(f"Kept {kept} reviews -> {reviews_out}")
    typer.echo("Data generation complete.")


if __name__ == "__main__":
    app()
