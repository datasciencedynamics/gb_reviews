"""Load the app's data artifacts. Missing files yield empty objects so the
page still renders with placeholders before the pipeline has run."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def default_data_dir() -> Path:
    env = os.environ.get("GB_APP_DATA")
    if env:
        return Path(env)
    local = HERE / "data"  # layout when copied into flask_apps
    if local.is_dir():
        return local
    return HERE.parent / "data" / "app"  # layout inside the gb_reviews repo


@dataclass
class AppData:
    summary: dict = field(default_factory=dict)
    changepoint: dict = field(default_factory=dict)
    aspects: pd.DataFrame = field(default_factory=pd.DataFrame)
    heat: pd.DataFrame = field(default_factory=pd.DataFrame)
    heat_single: pd.DataFrame = field(default_factory=pd.DataFrame)
    foods: pd.DataFrame = field(default_factory=pd.DataFrame)
    competitors: pd.DataFrame = field(default_factory=pd.DataFrame)
    fans: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def ready(self) -> bool:
        return bool(self.summary)


def _json(p: Path) -> dict:
    return json.loads(p.read_text()) if p.is_file() else {}


def _pq(p: Path) -> pd.DataFrame:
    return pd.read_parquet(p) if p.is_file() else pd.DataFrame()


def load(data_dir: Path | None = None) -> AppData:
    d = Path(data_dir) if data_dir else default_data_dir()
    return AppData(
        summary=_json(d / "summary.json"),
        changepoint=_json(d / "changepoint.json"),
        aspects=_pq(d / "aspects.parquet"),
        heat=_pq(d / "heat.parquet"),
        heat_single=_pq(d / "heat_single.parquet"),
        foods=_pq(d / "foods.parquet"),
        competitors=_pq(d / "competitors.parquet"),
        fans=_pq(d / "fans.parquet"),
    )
