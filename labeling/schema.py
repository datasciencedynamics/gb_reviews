"""Labeling prompt and the JSON schema the model must fill (via a forced tool)."""

from core.constants import (
    ASPECTS,
    COMPARISON_OUTCOMES,
    COMPETITORS,
    HEAT_DIRECTIONS,
    SAUCE_LABELS,
    SENTIMENTS,
)

SAUCE_KEYS = ["original", "green", "super_hot", "spicy_yellow"]

TOOL_NAME = "record_review_labels"

SYSTEM_PROMPT = f"""You annotate customer reviews of Gringo Bandito hot sauces for a \
research analysis. Read the review and record what the reviewer actually says. \
Never infer an opinion the reviewer did not express.

Sauces: {", ".join(f"{k} = {SAUCE_LABELS[k]}" for k in SAUCE_KEYS)}.

Aspects, each scored positive, negative, mixed, or not_mentioned:
- heat: how spicy it is, judged against what the reviewer wanted or expected.
- flavor: taste, aroma, ingredients, complexity.
- vinegar: vinegar or acidity specifically (a complaint about too much vinegar \
is negative; praise for low vinegar is positive).
- versatility: how many foods or uses it suits.
- value: price, size, shipping, packaging, value for money.

heat_direction says whether the reviewer found it milder than expected \
(too_mild), about right (as_expected), hotter than expected (too_hot), or did \
not discuss heat (not_mentioned). A reviewer who calls a "Super Hot" sauce \
"not really that hot" is too_mild even if they liked it.

sauces_discussed lists which sauces the review is about, using the keys above. \
The listing's sauce is given as context, but a variety-pack review may discuss \
several, and a review may name a different sauce than the listing.

foods: foods or dishes the reviewer uses it on, lowercase, singular \
("egg", "taco", "pizza").

competitors: other hot sauce brands the reviewer compares against, using these \
names when they match: {", ".join(COMPETITORS)}. Use the brand's own name \
otherwise. comparison is gb_better, gb_worse, similar, or unclear.

mentions_recipe_change: the reviewer says the recipe, heat or taste changed \
over time.
mentions_offspring: the reviewer mentions The Offspring or Dexter Holland.
mentions_reorder: the reviewer says they bought it again or will rebuy.
"""

_sentiment = {"type": "string", "enum": list(SENTIMENTS)}

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "aspects": {
            "type": "object",
            "properties": {a: _sentiment for a in ASPECTS},
            "required": list(ASPECTS),
        },
        "heat_direction": {"type": "string", "enum": list(HEAT_DIRECTIONS)},
        "sauces_discussed": {
            "type": "array",
            "items": {"type": "string", "enum": SAUCE_KEYS},
        },
        "peppers_mentioned": {"type": "array", "items": {"type": "string"}},
        "foods": {"type": "array", "items": {"type": "string"}},
        "competitors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "brand": {"type": "string"},
                    "comparison": {
                        "type": "string",
                        "enum": list(COMPARISON_OUTCOMES),
                    },
                },
                "required": ["brand", "comparison"],
            },
        },
        "mentions_recipe_change": {"type": "boolean"},
        "mentions_offspring": {"type": "boolean"},
        "mentions_reorder": {"type": "boolean"},
    },
    "required": [
        "aspects",
        "heat_direction",
        "sauces_discussed",
        "peppers_mentioned",
        "foods",
        "competitors",
        "mentions_recipe_change",
        "mentions_offspring",
        "mentions_reorder",
    ],
}

TOOL = {
    "name": TOOL_NAME,
    "description": "Record the structured labels for one review.",
    "input_schema": LABEL_SCHEMA,
}


def user_message(listing_sauce: str, rating: float, title: str, text: str) -> str:
    """Format one review for the model."""
    return (
        f"Listing sauce: {SAUCE_LABELS.get(listing_sauce, listing_sauce)}\n"
        f"Star rating: {rating:g}/5\n"
        f"Review title: {title}\n"
        f"Review text:\n{text}"
    )
