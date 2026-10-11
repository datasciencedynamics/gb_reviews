import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from labeling.io import attributed_sauces, load_labels
from modeling.aggregate import canonical_brand
from modeling.changepoint import best_split
from modeling.stats import wilson_ci
from preprocessing.data_gen import is_brand_item
from preprocessing.preprocessing import clean_text, map_sauce


@pytest.mark.parametrize("title,expected", [
    ("Gringo Bandito Super Hot Sauce, 5 Ounce (Pack of 4)", "super_hot"),
    ("Gringo Bandito Hot Sauce, Green Habanero, 5 Ounce", "green"),
    ("Gringo Bandito Original Red Hot Sauce 10oz", "original"),
    ("Gringo Bandito Spicy Yellow Hot Sauce", "spicy_yellow"),
    ("Gringo Bandito O.G. Hot Sauce Variety Pack", "variety"),
    ("Gringo Bandito Gift Crate", "variety"),
    ("Gringo Bandito Hot Sauce, 5 fl oz", "original"),
    ("Gringo Bandito T-Shirt", "unknown"),
])
def test_map_sauce(title, expected):
    assert map_sauce(title) == expected


def test_brand_match_and_clean():
    assert is_brand_item({"store": "Gringo Bandito", "title": "x"})
    assert is_brand_item({"title": "x", "details": {"Brand": "GRINGO BANDITO"}})
    assert not is_brand_item({"store": "Cholula", "title": "Hot Sauce"})
    assert clean_text("Great&amp;hot<br />sauce  ") == "Great&hot sauce"


def test_wilson_bounds():
    lo, hi = wilson_ci(5, 10)
    assert 0 < lo < 0.5 < hi < 1
    assert wilson_ci(0, 10)[0] == 0.0


def test_best_split_finds_planted_change():
    y = np.r_[np.zeros(100), np.ones(100)]
    idx, lr = best_split(y, min_seg=10)
    assert idx == 100 and lr > 50
    _, lr_flat = best_split(np.tile([0, 1], 100).astype(float), 10)
    assert lr_flat < lr


def test_attribution_and_brand_names():
    assert attributed_sauces("variety", ["green", "super_hot"]) == ["green", "super_hot"]
    assert attributed_sauces("green", []) == ["green"]
    assert attributed_sauces("variety", []) == []
    assert canonical_brand("tapatio") == "Tapatío"
    assert canonical_brand("Franks") == "Frank's RedHot"


def test_load_labels_keeps_success(synthetic):
    labels = load_labels(synthetic["primary"])
    reviews = pd.read_parquet(synthetic["reviews"])
    assert len(labels) == len(reviews)
    assert labels["heat"].notna().all()


def test_full_cli_chain(synthetic, monkeypatch):
    """validate -> changepoint -> aggregate on synthetic data, then the app."""
    from typer.testing import CliRunner

    from modeling import aggregate, changepoint, validate

    d = synthetic["dir"]
    runner = CliRunner()
    r = runner.invoke(validate.app, [
        "--reviews-file", str(synthetic["reviews"]), "--primary-labels", str(synthetic["primary"]),
        "--check-labels", str(synthetic["check"]), "--output-file", str(d / "validation.json"),
    ])
    assert r.exit_code == 0, r.output
    v = json.loads((d / "validation.json").read_text())
    assert v["agreement"]["fields"]["heat"]["kappa"] > 0.95
    assert v["agreement"]["fields"]["flavor"]["kappa"] < 1.0

    r = runner.invoke(changepoint.app, [
        "--reviews-file", str(synthetic["reviews"]), "--labels-file", str(synthetic["primary"]),
        "--n-permutations", "200", "--n-bootstraps", "100", "--output-file", str(d / "cp.json"),
    ])
    assert r.exit_code == 0, r.output
    cp = json.loads((d / "cp.json").read_text())
    assert cp["status"] == "ok"
    assert cp["p_value"] < 0.01
    assert "2018-06-01" < cp["change_date"] < "2019-07-01"
    assert cp["after"]["rate"] > cp["before"]["rate"] + 0.3

    out = d / "app"
    r = runner.invoke(aggregate.app, [
        "--reviews-file", str(synthetic["reviews"]), "--labels-file", str(synthetic["primary"]),
        "--validation-file", str(d / "validation.json"), "--changepoint-file", str(d / "cp.json"),
        "--output-dir", str(out),
    ])
    assert r.exit_code == 0, r.output
    summary = json.loads((out / "summary.json").read_text())
    assert summary["scopes"]["all"]["n"] == 1200
    heat = pd.read_parquet(out / "heat.parquet").set_index("sauce")
    assert heat.loc["super_hot", "too_mild"] > heat.loc["original", "too_mild"]
    comps = pd.read_parquet(out / "competitors.parquet")
    assert set(comps["brand"]) == {"Tapatío", "Cholula"}
    # review text never reaches the app folder
    for p in out.iterdir():
        if p.suffix == ".parquet":
            assert "text" not in pd.read_parquet(p).columns

    from app.app import load_data, overview_layout, technical_layout, build_app
    dash_app = build_app(data_dir=out)
    data = load_data(out)
    overview = str(overview_layout(dash_app, data, "/", "/technical"))
    technical = str(technical_layout(dash_app, data, "/", "/technical"))
    assert "Is Super Hot actually super hot?" in overview
    assert "Cohen's kappa" in technical and "heat-view" in technical
    assert "heat-view" not in overview  # technical controls stay off the overview
    single = pd.read_parquet(out / "heat_single.parquet").set_index("sauce")
    assert single.loc["super_hot", "too_mild"] > single.loc["original", "too_mild"]
    assert summary["heat_gap"]["single"]["fisher_p"] < 0.05


def test_app_renders_without_data(tmp_path):
    from app.app import build_app, load_data, overview_layout, technical_layout
    dash_app = build_app(data_dir=tmp_path)
    data = load_data(tmp_path)
    assert "No pipeline output found" in str(overview_layout(dash_app, data, "/", "/technical"))
    technical_layout(dash_app, data, "/", "/technical")  # renders with no data


def test_label_one_parses_tool_call():
    from labeling.label_reviews import label_one

    payload = {"aspects": {"heat": "negative"}, "heat_direction": "too_mild"}

    class FakeMessages:
        async def create(self, **kwargs):
            assert kwargs["tool_choice"]["type"] == "tool"
            return SimpleNamespace(content=[SimpleNamespace(type="tool_use", input=payload)])

    client = SimpleNamespace(messages=FakeMessages())
    row = {"review_id": "r1", "sauce": "super_hot", "rating": 4.0, "title": "t", "text": "not hot"}
    rec = asyncio.run(label_one(client, "m", row))
    assert rec["labels"] == payload and rec["review_id"] == "r1"
