
from pathlib import Path
import hashlib

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from clean_data import (
    TRAIN_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_clean_data,
    get_features_and_target,
)

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "mlflow.db"
EXPERIMENT_NAME = "MP122_v1"
MODEL_NAME = "MP122_Warehouse_Picking_Time"

# Use SQLite tracking in the repository root.
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
mlflow.set_experiment(EXPERIMENT_NAME)

# Required: hash the ORIGINAL raw training CSV before cleaning.
with open(TRAIN_PATH, "rb") as file:
    DATA_SHA256 = hashlib.sha256(file.read()).hexdigest()


def make_pipeline(model):
    numeric_features = [
        "items_in_order",
        "aisle_distance_m",
        "picker_exp_months",
        "shelf_height_m",
        "shift_hour",
        "order_weight_kg",
    ]

    categorical_features = ["zone_type", "shift"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def main():
    df = load_clean_data()
    X, y = get_features_and_target(df)

    # Keep the same validation rows for all three experiments.
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    experiments = [
        (
            "RandomForest_100",
            RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "RandomForest_200_depth10",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "ExtraTrees_200",
            ExtraTreesRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "GradientBoosting_100",
            GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=2,
                random_state=42,
            ),
        ),
    ]

    results = []

    for run_name, estimator in experiments:
        pipeline = make_pipeline(estimator)

        with mlflow.start_run(run_name=run_name) as run:
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_valid)
            mae = mean_absolute_error(y_valid, predictions)

            # Required parameters and metric in EVERY run.
            mlflow.log_param("data_sha256", DATA_SHA256)
            mlflow.log_param("model_type", run_name)
            mlflow.log_param("validation_fraction", 0.20)
            mlflow.log_param("random_state", 42)
            mlflow.log_metric("mae_validation", float(mae))

            # Save the fitted model with MLflow.
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                serialization_format="cloudpickle",
            )

            results.append({
                "run_id": run.info.run_id,
                "run_name": run_name,
                "mae": mae,
            })

            print(f"{run_name}: validation MAE = {mae:.4f}")
            print(f"Run ID: {run.info.run_id}")

    # Select the model with the lowest validation MAE.
    best = min(results, key=lambda result: result["mae"])

    print("\nBest model:", best["run_name"])
    print("Best validation MAE:", round(best["mae"], 4))

    # Register the best run's model.
    model_uri = f"runs:/{best['run_id']}/model"

    registered = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    # Assign the required champion alias.
    client = MlflowClient()
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=registered.version,
    )

    print("\nRegistered model:", MODEL_NAME)
    print("Model version:", registered.version)
    print("Champion alias: champion")

    print("\nAll experiment results:")
    for result in sorted(results, key=lambda item: item["mae"]):
        print(
            f"{result['run_name']}: "
            f"MAE = {result['mae']:.4f}"
        )


if __name__ == "__main__":
    main()
