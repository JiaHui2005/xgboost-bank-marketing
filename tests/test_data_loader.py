import pandas as pd
import pytest

from src.data_loader import encode_binary_target


def test_encode_binary_target_maps_expected_labels():
    encoded = encode_binary_target(pd.Series(["no", "yes", "no"]))

    assert encoded.tolist() == [0, 1, 0]
    assert str(encoded.dtype) == "int8"


@pytest.mark.parametrize(
    "labels",
    [["no", None], ["yes", "maybe"], ["no", float("nan")]],
)
def test_encode_binary_target_rejects_missing_or_unknown_labels(labels):
    with pytest.raises(ValueError, match="missing or unsupported labels"):
        encode_binary_target(pd.Series(labels))
