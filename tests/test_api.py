from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_valid_prediction():
    payload = {
        "items_in_order": 8,
        "aisle_distance_m": 104,
        "picker_exp_months": 56,
        "shelf_height_m": 1.09,
        "shift_hour": 23,
        "order_weight_kg": 13.43,
        "zone_type": "chilled",
        "shift": "Morning"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], float)


def test_invalid_experience_returns_422():
    payload = {
        "items_in_order": 8,
        "aisle_distance_m": 104,
        "picker_exp_months": 151,
        "shelf_height_m": 1.09,
        "shift_hour": 23,
        "order_weight_kg": 13.43,
        "zone_type": "chilled",
        "shift": "Morning"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_negative_weight_returns_422():
    payload = {
        "items_in_order": 8,
        "aisle_distance_m": 104,
        "picker_exp_months": 56,
        "shelf_height_m": 1.09,
        "shift_hour": 23,
        "order_weight_kg": -5,
        "zone_type": "chilled",
        "shift": "Morning"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
