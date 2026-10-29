"""Features available before the current marketing campaign begins."""

PRE_CAMPAIGN_FEATURES = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "pdays",
    "previous",
    "poutcome",
]

DROPPED_PRE_CAMPAIGN_FEATURES = [
    "duration",
    "campaign",
    "contact",
    "day",
    "month",
]
