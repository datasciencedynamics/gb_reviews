"""Small statistical helpers shared by validation, modeling and aggregation."""

import numpy as np
from scipy import stats

SENTIMENT_SCORE = {"positive": 1.0, "negative": -1.0, "mixed": 0.0}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def net_sentiment(sentiments) -> tuple:
    """(net score in [-1, 1], n mentions) over mentioned sentiments."""
    scores = [SENTIMENT_SCORE[s] for s in sentiments if s in SENTIMENT_SCORE]
    if not scores:
        return (float("nan"), 0)
    return (float(np.mean(scores)), len(scores))


def spearman(x, y) -> dict:
    """Spearman rho with p-value; NaN when there is too little data."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 10 or np.std(x) == 0 or np.std(y) == 0:
        return {"rho": float("nan"), "p": float("nan"), "n": int(len(x))}
    res = stats.spearmanr(x, y)
    return {"rho": float(res.statistic), "p": float(res.pvalue), "n": int(len(x))}
