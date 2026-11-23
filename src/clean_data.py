
from pathlib import Path
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"

# Exactly the 8 features accepted by the client API.
# Do not include record_id, date, pick_time_from_scanner or the target.
FEATURE_COLUMNS = [
    "items_in_order",
    "aisle_distance_m",
    "picker_exp_months",
    "shelf_height_m",
    "shift_hour",
    "order_weight_kg",
    "zone_type",
    "shift",
]

TARGET_COLUMN = "pick_seconds"


def load_clean_data(path=TRAIN_PATH):
    """Load and clean the original training data."""

    df = pd.read_csv(path)

    # Remove exact duplicate rows.
    df = df.drop_duplicates().copy()

    # Night-shift shelf heights were recorded in centimetres.
    # Convert them to metres.
    night_mask = df["shift"].astype(str).str.strip().str.lower() == "night"
    df.loc[night_mask, "shelf_height_m"] = (
        df.loc[night_mask, "shelf_height_m"] / 100
    )

    return df


def get_features_and_target(df):
    """Return model features and target, excluding leakage columns."""

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


if __name__ == "__main__":
    cleaned = load_clean_data()
    X, y = get_features_and_target(cleaned)

    print("Original training rows: 1056")
    print("Cleaned training rows:", len(cleaned))
    print("Features:", FEATURE_COLUMNS)
    print("Target:", TARGET_COLUMN)
    print("\nNight-shift shelf height after conversion:")
    print(cleaned.loc[cleaned["shift"] == "Night", "shelf_height_m"].describe())
