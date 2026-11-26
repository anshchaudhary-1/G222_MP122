import pandas as pd


def validate_dataset(file_path):
    df = pd.read_csv(file_path)

    print("\n--- Dataset Validation Report ---")

    print("\nDataset Shape:")
    print(df.shape)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Records:")
    print(df.duplicated().sum())

    print("\nData Types:")
    print(df.dtypes)

    print("\nNumeric Column Summary:")
    print(df.describe())

    print("\nTarget Variable Summary:")
    if "pick_seconds" in df.columns:
        print(df["pick_seconds"].describe())

    return df


if __name__ == "__main__":
    validate_dataset("data/train.csv") 