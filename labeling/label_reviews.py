#!/usr/bin/env python3
"""
label_reviews.py
----------------
Labels reviews with a Claude model through a forced tool call, so every
response is schema-valid JSON. Results are appended to a JSONL file one review
at a time; re-running skips reviews already labeled, so an interrupted run
resumes where it stopped.

Example usage:
    python labeling/label_reviews.py \
        --input-data-file ./data/processed/reviews.parquet \
        --output-file ./data/interim/labels_primary.jsonl \
        --model claude-haiku-5-5 \
        --sample-frac 1.0
"""

import asyncio
import json
import random
from pathlib import Path

import pandas as pd
import typer
from tqdm import tqdm

from core.config import rstate
from core.constants import LABEL_SCHEMA_VERSION, MAX_REVIEW_CHARS
from labeling.schema import SYSTEM_PROMPT, TOOL, TOOL_NAME, user_message

app = typer.Typer(add_completion=False)


def select_reviews(df: pd.DataFrame, sample_frac: float, seed: int = rstate) -> pd.DataFrame:
    """Deterministic sample: the same fraction always picks the same reviews."""
    if sample_frac >= 1.0:
        return df
    return df.sample(frac=sample_frac, random_state=seed).sort_values("date")


def already_labeled(path: Path) -> set:
    """Review ids with a successful label in an existing output file."""
    done = set()
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("labels") is not None:
                    done.add(rec["review_id"])
    return done


async def label_one(client, model: str, row: dict, max_retries: int = 6) -> dict:
    """Label one review, retrying transient API errors with backoff."""
    from anthropic import APIConnectionError

    text = row["text"][:MAX_REVIEW_CHARS]
    msg = user_message(row["sauce"], row["rating"], row["title"], text)
    tool_choice = {"type": "tool", "name": TOOL_NAME}
    for attempt in range(max_retries):
        try:
            resp = await client.messages.create(
                model=model,
                max_tokens=1024,
                system=SYSTEM_PROMPT
                + f"\nAlways respond by calling the {TOOL_NAME} tool.",
                tools=[TOOL],
                tool_choice=tool_choice,
                messages=[{"role": "user", "content": msg}],
            )
            block = next(
                (b for b in resp.content if getattr(b, "type", "") == "tool_use"),
                None,
            )
            if block is None:
                continue  # model answered without the tool; ask again
            return {
                "review_id": row["review_id"],
                "model": model,
                "schema_version": LABEL_SCHEMA_VERSION,
                "labels": block.input,
            }
        except Exception as e:  # noqa: BLE001 - surface any failure in the file
            status = getattr(e, "status_code", None)
            # Some models reject forced tool use; fall back to auto once
            if status == 400 and "tool_choice" in str(e) and tool_choice["type"] == "tool":
                tool_choice = {"type": "auto"}
                continue
            retryable = status in (408, 409, 429, 500, 502, 503, 504, 529) or isinstance(
                e, APIConnectionError
            )
            if attempt == max_retries - 1 or not retryable:
                return {
                    "review_id": row["review_id"],
                    "model": model,
                    "schema_version": LABEL_SCHEMA_VERSION,
                    "labels": None,
                    "error": f"{type(e).__name__}: {e}",
                }
            await asyncio.sleep(min(60, 2**attempt + random.random()))
    return {
        "review_id": row["review_id"],
        "model": model,
        "schema_version": LABEL_SCHEMA_VERSION,
        "labels": None,
        "error": "no tool call after retries",
    }



async def run(rows: list, model: str, output_file: Path, concurrency: int) -> tuple:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic()
    sem = asyncio.Semaphore(concurrency)
    ok, failed = 0, 0

    async def guarded(row):
        async with sem:
            return await label_one(client, model, row)

    with open(output_file, "a", encoding="utf-8") as f:
        tasks = [asyncio.create_task(guarded(r)) for r in rows]
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            rec = await fut
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if rec["labels"] is None:
                failed += 1
            else:
                ok += 1
    return ok, failed


@app.command()
def main(
    input_data_file: Path = typer.Option(
        Path("data/processed/reviews.parquet"), "--input-data-file"
    ),
    output_file: Path = typer.Option(
        Path("data/interim/labels_primary.jsonl"), "--output-file"
    ),
    model: str = typer.Option("claude-haiku-5-5", "--model"),
    sample_frac: float = typer.Option(1.0, "--sample-frac"),
    concurrency: int = typer.Option(8, "--concurrency"),
    limit: int = typer.Option(0, "--limit", help="0 = all; >0 labels only N (dry run)."),
) -> None:
    """Label reviews with structured aspect sentiment."""
    if not input_data_file.is_file():
        typer.secho(f"Input file not found: {input_data_file}", fg="red", err=True)
        raise typer.Exit(code=1)

    df = pd.read_parquet(input_data_file)
    df = select_reviews(df, sample_frac)
    done = already_labeled(output_file)
    todo = df[~df["review_id"].isin(done)]
    if limit > 0:
        todo = todo.head(limit)

    typer.echo(
        f"Model {model}: {len(df):,} selected, {len(done):,} already labeled, "
        f"{len(todo):,} to label"
    )
    if todo.empty:
        typer.echo("Nothing to do.")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    rows = todo[["review_id", "sauce", "rating", "title", "text"]].to_dict("records")
    ok, failed = asyncio.run(run(rows, model, output_file, concurrency))
    typer.echo(f"Labeled {ok:,}, failed {failed:,} -> {output_file}")
    if failed:
        typer.echo("Re-run the same command to retry failures.")


if __name__ == "__main__":
    app()
