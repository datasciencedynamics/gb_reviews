"""Synthetic fixtures for tests only. Nothing here is real review data and
none of it is ever written to data/ or shown in the app."""

import json

import numpy as np
import pandas as pd
import pytest

SAUCES = ["original", "green", "super_hot", "spicy_yellow"]
TITLES = {
    "original": "Gringo Bandito Hot Sauce, Original Red, 5 Ounce (Pack of 4)",
    "green": "Gringo Bandito Hot Sauce, Green Habanero, 5 Ounce (Pack of 4)",
    "super_hot": "Gringo Bandito Super Hot Sauce, 5 Ounce (Pack of 4)",
    "spicy_yellow": "Gringo Bandito Spicy Yellow Hot Sauce, 5 oz",
}


@pytest.fixture
def synthetic(tmp_path):
    """Planted effect: Super Hot 'too mild' rate jumps from 10% to 60% in 2019."""
    rng = np.random.default_rng(0)
    meta = pd.DataFrame(
        [{"parent_asin": f"P{i}", "title": TITLES[s], "store": "Gringo Bandito"}
         for i, s in enumerate(SAUCES)]
        + [{"parent_asin": "P9", "title": "Gringo Bandito O.G. Hot Sauce Variety Pack",
            "store": "Gringo Bandito"}]
    )
    rows, labels = [], []
    start = pd.Timestamp("2012-01-01")
    for i in range(1200):
        k = i % 5
        asin = f"P{k}" if k < 4 else "P9"
        sauce = SAUCES[k] if k < 4 else None
        date = start + pd.Timedelta(days=int(rng.integers(0, 365 * 11)))
        rows.append({
            "parent_asin": asin, "asin": asin, "user_id": f"U{i}",
            "timestamp": int(date.timestamp() * 1000),
            "rating": float(rng.integers(1, 6)), "title": f"t{i}",
            "text": f"synthetic review {i}", "verified_purchase": bool(rng.random() < 0.9),
            "helpful_vote": 0,
        })
        after = date >= pd.Timestamp("2019-01-01")
        p_mild = 0.6 if (sauce == "super_hot" and after) else 0.1
        heat_dir = "too_mild" if rng.random() < p_mild else rng.choice(["as_expected", "too_hot"])
        labels.append({
            "heat": "negative" if heat_dir == "too_mild" else "positive",
            "flavor": rng.choice(["positive", "negative", "not_mentioned"]),
            "vinegar": rng.choice(["negative", "not_mentioned"]),
            "versatility": "positive", "value": "not_mentioned",
            "heat_direction": heat_dir,
            "sauces_discussed": [sauce] if sauce else ["original", "green"],
            "peppers_mentioned": ["scorpion"] if rng.random() < 0.05 else [],
            "foods": [rng.choice(["egg", "taco", "pizza"])],
            "competitors": [{"brand": rng.choice(["tapatio", "Cholula"]),
                             "comparison": rng.choice(["gb_better", "gb_worse"])}],
            "mentions_recipe_change": bool(rng.random() < 0.03),
            "mentions_offspring": bool(rng.random() < 0.1),
            "mentions_reorder": bool(rng.random() < 0.3),
        })

    from preprocessing.preprocessing import build_reviews
    reviews = build_reviews(meta, pd.DataFrame(rows))

    def write_labels(path, model, frac=1.0, flip=0.0):
        r = np.random.default_rng(1)
        with open(path, "w") as f:
            # build_reviews sorts by date, so align labels by the row's user id
            for rid, uid in zip(reviews["review_id"], reviews["user_id"]):
                lab = labels[int(uid[1:])]
                if r.random() > frac:
                    continue
                lab = dict(lab)
                if r.random() < flip:
                    lab["flavor"] = "mixed"
                aspects = {a: lab.pop(a) for a in ("heat", "flavor", "vinegar", "versatility", "value")}
                f.write(json.dumps({"review_id": rid, "model": model, "schema_version": "v1",
                                    "labels": {"aspects": aspects, **lab}}) + "\n")
                # a failed attempt on the same id must not override success
                if r.random() < 0.01:
                    f.write(json.dumps({"review_id": rid, "model": model,
                                        "labels": None, "error": "x"}) + "\n")

    reviews_path = tmp_path / "reviews.parquet"
    reviews.to_parquet(reviews_path, index=False)
    primary = tmp_path / "labels_primary.jsonl"
    check = tmp_path / "labels_check.jsonl"
    write_labels(primary, "primary-model")
    write_labels(check, "check-model", frac=0.2, flip=0.1)
    return {"dir": tmp_path, "reviews": reviews_path, "primary": primary, "check": check}
