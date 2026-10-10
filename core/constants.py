################################################################################
############################## Brand Matching ##################################
################################################################################

# Case-insensitive substrings matched against a product's store, title and
# brand fields in the Amazon metadata.
BRAND_KEYS = ("gringo bandito",)

################################################################################
################################# Sauces #######################################
################################################################################

# Ordered so that more specific patterns win (a variety pack title can name
# several sauces; it is mapped to "variety"). Patterns are lowercase.
SAUCE_PATTERNS = {
    "super_hot": ("super hot", "superhot"),
    "spicy_yellow": ("spicy yellow", "yellow"),
    "green": ("green habanero", "green"),
    "original": ("original red", "original", "red hot sauce"),
}

SAUCE_LABELS = {
    "original": "Original Red",
    "green": "Green Habanero",
    "super_hot": "Super Hot",
    "spicy_yellow": "Spicy Yellow",
    "variety": "Variety / mixed",
    "unknown": "Unknown",
}

SAUCE_COLORS = {
    "original": "#D7392B",
    "green": "#4E9A55",
    "super_hot": "#EE7A2B",
    "spicy_yellow": "#E3B23C",
    "variety": "#8C8378",
    "unknown": "#6B6157",
}

# Child ASINs under the pooled "Super Hot" parent listing, checked on Amazon 2026-10-10.
# "variety" means flavor comes from the review text, not the listing.
CHILD_ASIN_SAUCE = {
    "B00XLHS8HQ": "variety",   # O.G. Variety Pack
    "B00XLHS9EI": "variety",   # "Super Hot Sauce Variety Pack" (ambiguous title)
    "B00CXZKCTY": "variety",   # Variety Pack
    "B00U9VXRQY": "variety",   # Classic Variety Pack
    "B009WPKGQC": "original",  # Original Red 4-pack
    "B00C4TKL4A": "variety",   # dead page
    "B00CXZKD16": "green",     # Green Habanero 4-pack
    "B00XLHS85S": "variety",   # not checked
    "B07Q4819YL": "variety",   # not checked
    "B07Q9YQX1T": "variety",   # not checked
    "B07QX8RBGB": "variety",   # not checked
    "B00DOAZA8U": "variety",   # not checked
}

# Scoville values printed on the brand's store (checked 2026-10-09) and
# earlier test results reported by Pepper Geek. None = not reported.
LABELED_SHU = {
    "original": 1590,
    "green": 1670,
    "super_hot": 3530,
    "spicy_yellow": 3600,
}
EARLIER_SHU = {
    "original": 820,
    "green": None,
    "super_hot": 2900,
    "spicy_yellow": None,
}
SHU_SOURCES = {
    "labeled": "gringobanditostore.com product pages, checked 2026-10-09",
    "earlier": "Pepper Geek, Gringo Bandito Hot Sauce Review",
}

# HPLC results go here once current bottles are lab tested.
LAB_SHU = {
    "original": None,
    "green": None,
    "super_hot": None,
    "spicy_yellow": None,
}

################################################################################
################################# Aspects ######################################
################################################################################

ASPECTS = ("heat", "flavor", "vinegar", "versatility", "value")

ASPECT_LABELS = {
    "heat": "Heat",
    "flavor": "Flavor",
    "vinegar": "Vinegar",
    "versatility": "Versatility",
    "value": "Value",
}

SENTIMENTS = ("positive", "negative", "mixed", "not_mentioned")

# Direction of a heat mention relative to what the reviewer expected
HEAT_DIRECTIONS = ("too_mild", "as_expected", "too_hot", "not_mentioned")

COMPETITORS = (
    "Cholula",
    "Tapatío",
    "Tabasco",
    "Valentina",
    "Huy Fong",
    "Trader Joe's",
    "Frank's RedHot",
    "Secret Aardvark",
    "El Yucateco",
)

COMPARISON_OUTCOMES = ("gb_better", "gb_worse", "similar", "unclear")

################################################################################
############################### Labeling ######################################
################################################################################

LABEL_SCHEMA_VERSION = "v1"
MAX_REVIEW_CHARS = 4000  # truncate very long reviews before labeling
