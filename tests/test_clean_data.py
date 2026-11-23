
import sys
from pathlib import Path

import pandas as pd

# Make the src folder importable during tests.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clean_data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_clean_data,
    get_features_and_target,
)


def test_duplicate_rows_are_removed():
    df = load_clean_data()
    assert len(df) == 1000
    assert not df.duplicated().any()


def test_night_shelf_heights_are_converted():
    df = load_clean_data()
    night = df[df["shift"] == "Night"]

    assert night["shelf_height_m"].max() <= 3
    assert night["shelf_height_m"].min() >= 0.3


def test_feature_columns_match_api():
    expected = [
        "items_in_order",
        "aisle_distance_m",
        "picker_exp_months",
        "shelf_height_m",
        "shift_hour",
        "order_weight_kg",
        "zone_type",
        "shift",
    ]

    assert FEATURE_COLUMNS == expected


def test_leakage_and_identifiers_are_excluded():
    df = load_clean_data()
    X, y = get_features_and_target(df)

    assert "pick_time_from_scanner" not in X.columns
    assert "record_id" not in X.columns
    assert "date" not in X.columns
    assert TARGET_COLUMN not in X.columns
    assert len(X) == len(y)


def test_target_is_pick_seconds():
    df = load_clean_data()
    X, y = get_features_and_target(df)

    assert y.name == "pick_seconds"
    assert len(X.columns) == 8
