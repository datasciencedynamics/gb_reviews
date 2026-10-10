#!/usr/bin/env python3
"""
changepoint.py
--------------
Did a sauce start reading as milder? Among reviews of one sauce that discuss
heat, the outcome is whether the reviewer found it milder than expected
(heat_direction == "too_mild"). A single change point in that Bernoulli rate
is located by maximum likelihood over review dates.

- Significance: permutation test on the likelihood-ratio statistic (shuffling
  outcomes over dates is valid under the no-change null).
- Uncertainty in the date: bootstrap percentile interval.
- Effect: rates before and after with Wilson intervals, plus mean stars.
- Context: yearly counts of reviews that mention a recipe change or a scorpion
  pepper, which should move with any real reformulation.

Example usage:
    python modeling/changepoint.py \
        --reviews-file ./data/processed/reviews.parquet \
        --labels-file ./data/interim/labels_primary.jsonl \
        --sauce super_hot \
        --output-file ./reports/eval/changepoint_super_hot.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from core.config import rstate
from core.constants import SAUCE_LABELS
from labeling.io import join_reviews_labels, load_labels
from modeling.stats import wilson_ci

app = typer.Typer(add_completion=False)


def _loglik(k: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Bernoulli log-likelihood at the MLE, vectorised; 0*log0 treated as 0."""
    p = np.divide(k, n, out=np.zeros_like(k, dtype=float), where=n > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(k > 0, k * np.log(p), 0.0)
        b = np.where(n - k > 0, (n - k) * np.log1p(-p), 0.0)
    return a + b


def best_split(y: np.ndarray, min_seg: int) -> tuple:
    """Index i (split before y[i]) maximising the two-rate likelihood, and
    the likelihood-ratio statistic 2*(ll_split - ll_null)."""
    n = len(y)
    c = np.cumsum(y)
    idx = np.arange(min_seg, n - min_seg + 1)
    if len(idx) == 0:
        return None, 0.0
    k1, n1 = c[idx - 1], idx.astype(float)
    k2, n2 = c[-1] - k1, n - n1
    ll = _loglik(k1, n1) + _loglik(k2, n2)
    j = int(np.argmax(ll))
    ll_null = float(_loglik(np.array([c[-1]]), np.array([float(n)]))[0])
    return int(idx[j]), float(2 * (ll[j] - ll_null))


def detect(df: pd.DataFrame, min_seg: int, n_perm: int, n_boot: int, seed: int) -> dict:
    """Change point on a date-sorted frame with a boolean 'y' column."""
    df = df.sort_values("date").reset_index(drop=True)
    y = df["y"].to_numpy(dtype=float)
    split, lr = best_split(y, min_seg)
    if split is None:
        return {"status": "insufficient_data", "n": int(len(df)), "min_segment": min_seg}

    rng = np.random.default_rng(seed)
    perm = np.array([best_split(rng.permutation(y), min_seg)[1] for _ in range(n_perm)])
    p_value = float((1 + (perm >= lr).sum()) / (1 + n_perm))

    boot_dates = []
    for _ in range(n_boot):
        b = df.iloc[rng.integers(0, len(df), len(df))].sort_values("date")
        s, _ = best_split(b["y"].to_numpy(dtype=float), min_seg)
        if s is not None:
            boot_dates.append(b["date"].iloc[s].value)
    lo, hi = (
        pd.to_datetime(np.percentile(boot_dates, [2.5, 97.5])) if boot_dates else (None, None)
    )

    before, after = df.iloc[:split], df.iloc[split:]

    def seg(d):
        k, n = int(d["y"].sum()), len(d)
        ci = wilson_ci(k, n)
        return {
            "n": n,
            "k": k,
            "rate": k / n if n else float("nan"),
            "ci": list(ci),
            "mean_rating": float(d["rating"].mean()),
            "start": str(d["date"].min().date()),
            "end": str(d["date"].max().date()),
        }

    return {
        "status": "ok",
        "n": int(len(df)),
        "min_segment": min_seg,
        "change_date": str(df["date"].iloc[split].date()),
        "change_date_ci": [str(lo.date()), str(hi.date())] if lo is not None else None,
        "lr_statistic": lr,
        "p_value": p_value,
        "n_permutations": n_perm,
        "n_bootstraps": n_boot,
        "before": seg(before),
        "after": seg(after),
    }


def rolling_series(df: pd.DataFrame, window_months: int = 6) -> list:
    """Monthly points of a trailing-window rate with Wilson intervals."""
    m = df.groupby("month")["y"].agg(["sum", "count"]).sort_index()
    if m.empty:
        return []
    full = pd.date_range(m.index.min(), m.index.max(), freq="MS")
    m = m.reindex(full, fill_value=0)
    k = m["sum"].rolling(window_months, min_periods=1).sum()
    n = m["count"].rolling(window_months, min_periods=1).sum()
    out = []
    for month, ki, ni in zip(m.index, k, n):
        if ni < 5:
            continue  # too thin to plot honestly
        lo, hi = wilson_ci(int(ki), int(ni))
        out.append(
            {"month": str(month.date()), "rate": ki / ni, "lo": lo, "hi": hi, "n": int(ni)}
        )
    return out


def mention_timeline(df: pd.DataFrame) -> list:
    """Yearly counts of recipe-change and scorpion mentions."""
    d = df.assign(
        year=df["date"].dt.year,
        scorpion=df["peppers_mentioned"].map(lambda ps: any("scorpion" in p for p in ps)),
    )
    g = d.groupby("year").agg(
        reviews=("review_id", "size"),
        recipe_change=("mentions_recipe_change", "sum"),
        scorpion=("scorpion", "sum"),
    )
    return [{"year": int(y), **{k: int(v) for k, v in r.items()}} for y, r in g.iterrows()]


@app.command()
def main(
    reviews_file: Path = typer.Option(Path("data/processed/reviews.parquet"), "--reviews-file"),
    labels_file: Path = typer.Option(Path("data/interim/labels_primary.jsonl"), "--labels-file"),
    sauce: str = typer.Option("super_hot", "--sauce"),
    min_segment: int = typer.Option(15, "--min-segment", help="Fewest heat reviews per side."),
    n_permutations: int = typer.Option(1000, "--n-permutations"),
    n_bootstraps: int = typer.Option(1000, "--n-bootstraps"),
    output_file: Path = typer.Option(Path("reports/eval/changepoint_super_hot.json"), "--output-file"),
) -> None:
    """Locate and test a change in 'milder than expected' heat reviews."""
    df = join_reviews_labels(pd.read_parquet(reviews_file), load_labels(labels_file))
    df = df[df["sauces"].map(lambda s: sauce in s)]
    heat = df[df["heat_direction"] != "not_mentioned"].copy()
    heat["y"] = heat["heat_direction"] == "too_mild"

    typer.echo(
        f"{SAUCE_LABELS.get(sauce, sauce)}: {len(df):,} reviews, "
        f"{len(heat):,} discuss heat, {int(heat['y'].sum())} call it milder than expected"
    )
    result = detect(heat, min_segment, n_permutations, n_bootstraps, rstate)
    result.update(
        sauce=sauce,
        outcome="heat_direction == too_mild, among reviews that discuss heat",
        series=rolling_series(heat),
        mentions=mention_timeline(df),
    )

    if result["status"] == "ok":
        b, a = result["before"], result["after"]
        typer.echo(
            f"\nChange point: {result['change_date']} "
            f"(95% bootstrap interval {result['change_date_ci'][0]} to {result['change_date_ci'][1]})"
        )
        typer.echo(f"Permutation p = {result['p_value']:.4f}, LR = {result['lr_statistic']:.2f}")
        typer.echo(f"Before: {b['rate']:.1%} milder than expected (n={b['n']}), mean stars {b['mean_rating']:.2f}")
        typer.echo(f"After:  {a['rate']:.1%} milder than expected (n={a['n']}), mean stars {a['mean_rating']:.2f}")
    else:
        typer.echo(f"Not enough heat reviews for a change-point fit (need {2 * min_segment}).")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, indent=2, default=float))
    typer.echo(f"\nSaved {output_file}")


if __name__ == "__main__":
    app()
