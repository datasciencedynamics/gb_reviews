# Gringo Bandito Review Lab

Aspect-level analysis of every Amazon review of Gringo Bandito hot sauces in the
McAuley Lab **Amazon Reviews'23** release (through September 2023), plus a Dash
app that presents it.

Centerpiece question: **did Super Hot start reading as milder?** Reviews that
discuss heat are labeled for whether the sauce felt milder than expected, and a
change-point model locates and tests any shift.

## Pipeline

```
make create_folders
make requirements
cp .env.example .env              # add ANTHROPIC_API_KEY

make data_gen                     # stream the dataset, keep Gringo Bandito rows (slow, once)
make preprocess                   # clean, map listings to sauces -> data/processed/reviews.parquet
make label_primary LIMIT=25       # dry run on 25 reviews, inspect data/interim/labels_primary.jsonl
make label_pipeline               # label everything, then relabel a 10% sample with a second model
make model_pipeline               # validation, change point, app exports
make app                          # http://localhost:8050
```

`make help` lists every target. Labeling is resumable: re-running skips reviews
already labeled and retries failures.

## What each step does

| Step | Script | Output |
|---|---|---|
| Pull | `preprocessing/data_gen.py` | `data/raw/gb_meta.jsonl`, `gb_reviews.jsonl` |
| Clean | `preprocessing/preprocessing.py` | `data/processed/reviews.parquet` (prints listing-to-sauce mapping for review) |
| Label | `labeling/label_reviews.py` | `data/interim/labels_*.jsonl` |
| Validate | `modeling/validate.py` | `reports/eval/validation.json` |
| Change point | `modeling/changepoint.py` | `reports/eval/changepoint_super_hot.json` |
| Export | `modeling/aggregate.py` | `data/app/*` (counts and rates only, no review text) |

## Methods

- **Labels.** A Claude model fills a fixed schema through a forced tool call:
  sentiment for heat, flavor, vinegar, versatility and value; heat direction
  (milder than expected, as expected, hotter); foods; competitor comparisons;
  mentions of a recipe change, The Offspring, and reordering.
- **Validation without hand coding.** A second model relabels a random 10%;
  agreement is Cohen's kappa per field. Aspect sentiment is also checked
  against star ratings (Spearman).
- **Change point.** Bernoulli rate of "milder than expected" among heat reviews
  of one sauce, split at the maximum-likelihood date. Permutation test for
  significance, bootstrap percentile interval for the date, Wilson intervals
  for the rates before and after.
- **Suppression.** Cells built on fewer than five reviews show as "too few".

## Data and terms

Amazon Reviews'23 is an academic release for non-commercial research. Cite:
Hou et al. (2024), *Bridging Language and Items for Retrieval and
Recommendation*, arXiv:2403.03952. The app shows aggregates only.

Scoville values in `core/constants.py` come from the brand's product pages
(checked 2026-10-09) and Pepper Geek's review. Add HPLC results to `LAB_SHU`
once current bottles are lab tested.

## Deploying in flask_apps

```
make export_flask_apps FLASK_APPS=~/Python_Projects/flask_apps
```

then in the launcher:

```python
from gb_reviews.app import create_app
create_app(server=server, url_base_pathname="/gb-reviews/")
```

## Tests

`make test` runs the full chain on a synthetic fixture with a planted change
point. Synthetic data lives only in the tests and is never written to `data/`.
