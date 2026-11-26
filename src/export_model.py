
from pathlib import Path
import mlflow
import mlflow.sklearn
import joblib

ROOT = Path(__file__).resolve().parents[1]

# Connect to the existing MLflow database.
mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")

# Load the registered model using its champion alias.
model_uri = "models:/MP122_Warehouse_Picking_Time@champion"
model = mlflow.sklearn.load_model(model_uri)

# Export the fitted pipeline for the FastAPI application.
output_path = ROOT / "models" / "champion_model.joblib"
joblib.dump(model, output_path)

print("Model exported successfully!")
print("File:", output_path)
print("Size:", round(output_path.stat().st_size / (1024 * 1024), 2), "MB")
