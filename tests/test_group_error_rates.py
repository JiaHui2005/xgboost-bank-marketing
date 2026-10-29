import pandas as pd

from scripts.analyze_group_error_rates import add_group_error_rates


def test_group_false_positive_metrics_use_distinct_denominators():
    summary = pd.DataFrame(
        {
            "total": [10],
            "actual_positive": [4],
            "false_positive": [3],
            "false_negative": [1],
        }
    )

    result = add_group_error_rates(summary).iloc[0]

    # FPR = FP / (FP + TN) = 3 / 6; group fraction = FP / all records.
    assert result["actual_negative"] == 6
    assert result["false_positive_rate"] == 0.5
    assert result["false_positive_fraction_within_group"] == 0.3


def test_group_false_positive_rate_is_undefined_without_actual_negatives():
    summary = pd.DataFrame(
        {
            "total": [4],
            "actual_positive": [4],
            "false_positive": [0],
            "false_negative": [1],
        }
    )

    result = add_group_error_rates(summary).iloc[0]

    assert pd.isna(result["false_positive_rate"])
    assert result["false_positive_fraction_within_group"] == 0


def test_group_false_negative_rate_is_undefined_without_actual_positives():
    summary = pd.DataFrame(
        {
            "total": [4],
            "actual_positive": [0],
            "false_positive": [1],
            "false_negative": [0],
        }
    )

    result = add_group_error_rates(summary).iloc[0]

    assert pd.isna(result["false_negative_rate_within_actual_positive"])
    assert pd.isna(result["missed_positive_share"])
