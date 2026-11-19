
import pandas as pd
from pathlib import Path

# Locate the project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"

# Load training dataset
df = pd.read_csv(TRAIN_PATH)

print("=" * 60)
print("MP122 - WAREHOUSE PICKING TIME DATA INSPECTION")
print("=" * 60)

# 1. Dataset dimensions
print("\n1. DATASET SHAPE")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 2. Column information
print("\n2. COLUMN INFORMATION")
print(df.info())

# 3. Missing values
print("\n3. MISSING VALUES")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "No missing values found.")

# 4. Duplicate rows
print("\n4. DUPLICATE ROWS")
print("Duplicate rows:", df.duplicated().sum())

# 5. Numerical statistics
print("\n5. NUMERICAL STATISTICS")
print(df.describe().to_string())

# 6. Categorical values
print("\n6. CATEGORICAL VALUES")
for column in df.select_dtypes(include="object").columns:
    print(f"\n{column}:")
    print(df[column].value_counts(dropna=False).to_string())

print("\nBasic inspection completed.")






print("\n" + "=" * 60)
print("DETAILED DATA QUALITY INVESTIGATION")
print("=" * 60)

# 1. Duplicate record IDs
print("\n1. DUPLICATE RECORD IDs")
duplicate_ids = df[df["record_id"].duplicated(keep=False)]
print("Rows with repeated IDs:", len(duplicate_ids))
print(duplicate_ids.sort_values("record_id").to_string(index=False))

# 2. Suspicious shelf heights
print("\n2. SUSPICIOUS SHELF HEIGHTS")
print(df[df["shelf_height_m"] > 10][
    ["record_id", "shelf_height_m"]
].to_string(index=False))

# 3. Compare scanner time with target
print("\n3. SCANNER TIME VS TARGET")
df["scanner_target_difference"] = (
    df["pick_time_from_scanner"] - df["pick_seconds"]
)
print(df["scanner_target_difference"].describe())
print("Exact matches:", (
    df["pick_time_from_scanner"] == df["pick_seconds"]
).sum())

# 4. Check target outliers
print("\n4. HIGHEST PICK TIMES")
print(
    df.nlargest(20, "pick_seconds")[
        ["record_id", "pick_seconds", "pick_time_from_scanner"]
    ].to_string(index=False)
)

# 5. Check categorical spelling variants
print("\n5. CATEGORICAL SPELLING CHECK")
for column in ["zone_type", "shift"]:
    print(f"\n{column}:")
    print(df[column].value_counts(dropna=False).to_string())
    
    
    
    
    
    
    
    
    
    
    
    # 6. SENTINEL VALUE CHECK
print("\n6. SENTINEL VALUE CHECK")

numeric_cols = df.select_dtypes(include="number").columns

for col in numeric_cols:
    print(f"\n{col}")
    print("Zero values:", (df[col] == 0).sum())
    print("Negative values:", (df[col] < 0).sum())
    print("Most frequent values:")
    print(df[col].value_counts().head(10).to_string())


# 7. STUCK SENSOR CHECK
print("\n7. STUCK SENSOR CHECK")

sensor = df["pick_time_from_scanner"]

longest_run = 1
current_run = 1

for i in range(1, len(sensor)):
    if sensor.iloc[i] == sensor.iloc[i - 1]:
        current_run += 1
        longest_run = max(longest_run, current_run)
    else:
        current_run = 1

print("Longest consecutive identical scanner reading:", longest_run)


# 8. TARGET OUTLIER CHECK
print("\n8. TARGET OUTLIER CHECK")

q1 = df["pick_seconds"].quantile(0.25)
q3 = df["pick_seconds"].quantile(0.75)
iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outliers = df[
    (df["pick_seconds"] < lower) |
    (df["pick_seconds"] > upper)
]

print("IQR lower limit:", lower)
print("IQR upper limit:", upper)
print("Potential target outliers:", len(outliers))
print(outliers[
    ["record_id", "pick_time_from_scanner", "pick_seconds"]
].to_string(index=False))


# 9. SHELF HEIGHT SUMMARY
print("\n9. SHELF HEIGHT SUMMARY")

suspicious = df[df["shelf_height_m"] > 10]

print("Potential centimetre entries:", len(suspicious))
print("Percentage of dataset:",
      round(len(suspicious) / len(df) * 100, 2), "%")








print("\n10. ZERO EXPERIENCE INVESTIGATION")

zero_exp = df[df["picker_exp_months"] == 0]

print("Number of records:", len(zero_exp))

print(zero_exp[
    ["record_id", "date", "picker_exp_months",
     "pick_seconds", "shift", "zone_type"]
].to_string(index=False)) 











print("\n11. UNIT MIX-UP BY ZONE")

for zone, group in df.groupby("zone_type"):
    print(f"\nZone: {zone}")
    print("Count:", len(group))
    print("Minimum:", group["shelf_height_m"].min())
    print("Mean:", round(group["shelf_height_m"].mean(), 2))
    print("Maximum:", group["shelf_height_m"].max())

print("\n12. UNIT MIX-UP BY SHIFT")

for shift, group in df.groupby("shift"):
    heights = group["shelf_height_m"]

    print(f"\nShift: {shift}")
    print("Count:", len(group))
    print("Minimum:", heights.min())
    print("Mean:", round(heights.mean(), 2))
    print("Maximum:", heights.max())
    print("Values above 10:", (heights > 10).sum()) 
    
    
    
    
    
    
    
    
    
    

print("\n13. STUCK SENSOR INVESTIGATION")

sensor_df = df.sort_values("date").copy()

col = "pick_time_from_scanner"

# Start a new run whenever the sensor reading changes
sensor_df["run_id"] = (
    sensor_df[col] != sensor_df[col].shift(1)
).cumsum()

runs = sensor_df.groupby("run_id").agg(
    reading=(col, "first"),
    count=(col, "size"),
    first_date=("date", "min"),
    last_date=("date", "max")
)

runs = runs.sort_values("count", ascending=False)

print("Top 10 consecutive identical readings:")
print(runs.head(10).to_string())

print("\nLongest run:")
print(runs.iloc[0].to_string())




print("\n14. TARGET TYPO INVESTIGATION")

target_sorted = df.sort_values(
    "pick_seconds",
    ascending=False
)

print("Top 30 target values:")
print(
    target_sorted[
        ["record_id", "pick_seconds", "pick_time_from_scanner"]
    ].head(30).to_string(index=False)
)


print("\nLargest gaps in sorted target values:")

ascending = (
    df["pick_seconds"]
    .sort_values()
    .reset_index(drop=True)
)

gaps = ascending.diff()

largest_gap_indices = (
    gaps.nlargest(10).index
)

for idx in largest_gap_indices:
    print(
        f"Below: {ascending.iloc[idx-1]}, "
        f"Above: {ascending.iloc[idx]}, "
        f"Gap: {gaps.iloc[idx]}"
    )




print("\n15. SPELLING VARIANT INVESTIGATION")

for col in ["zone_type", "shift"]:
    print(f"\nColumn: {col}")
    print("Unique values:")
    print(df[col].value_counts(dropna=False).to_string())
    
print("\n16. SENTINEL VALUE INVESTIGATION")

numeric_cols = [
    "items_in_order",
    "aisle_distance_m",
    "picker_exp_months",
    "shelf_height_m",
    "shift_hour",
    "order_weight_kg",
    "pick_time_from_scanner",
    "pick_seconds"
]

for col in numeric_cols:
    print(f"\n{col}")
    print("Smallest values:")
    print(df[col].value_counts().sort_index().head(5))

    print("Largest values:")
    print(df[col].value_counts().sort_index().tail(5))
    