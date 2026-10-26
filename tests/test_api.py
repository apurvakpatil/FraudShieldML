"""
Comprehensive Unit and Integration Tests for Credit Card Fraud Detection System
Tests data discovery, preprocessing pipeline, model inference, and Flask REST API endpoints.
"""

import os
import sys
import pytest
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app
from src.data_loader import find_dataset_path, get_sample_transactions
from src.predict import get_predictor


@pytest.fixture
def client():
    """Configures a Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_dataset_discovery():
    """Verifies that the original uploaded dataset is found."""
    path = find_dataset_path()
    assert os.path.isfile(path), f"Dataset file does not exist at: {path}"
    assert path.lower().endswith(".csv"), "Dataset is not a CSV file."


def test_sample_transactions():
    """Verifies genuine transaction extraction from the dataset."""
    samples = get_sample_transactions()
    assert len(samples) > 0, "No sample transactions extracted."
    for s in samples:
        assert "amt" in s
        assert "category" in s
        assert "is_fraud" in s
        assert s["is_fraud"] in [0, 1]


def test_model_loaded():
    """Verifies that the saved model artifact is loaded and ready."""
    predictor = get_predictor()
    assert predictor.is_loaded, "Model pipeline failed to load from models/fraud_detection_model.joblib"
    assert predictor.model_name is not None
    assert predictor.threshold > 0.0


def test_health_endpoint(client):
    """Tests GET /api/health."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["production_model"] == "Gradient Boosting"
    assert "model_selection_strategy" in data


def test_prediction_legitimate(client):
    """Tests POST /api/predict with standard low-risk transaction (automatic model selection)."""
    payload = {
        "amt": 25.50,
        "category": "grocery_pos",
        "hour": 14,
        "day_of_week": 2,
        "distance_km": 5.2,
        "age": 35,
        "gender": "M",
        "city_pop": 50000
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "prediction" in data
    assert "fraud_probability" in data
    assert "score" in data
    assert data["score"] == data["fraud_probability"]
    assert data["model_name"] == "Gradient Boosting"
    assert data["selected_model"] == "Gradient Boosting"
    assert "threshold" in data
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["prediction"] in [0, 1]


def test_prediction_fraud_scenario(client):
    """Tests POST /api/predict with anomalous high-risk attributes."""
    payload = {
        "amt": 1250.00,
        "category": "shopping_net",
        "hour": 23,
        "day_of_week": 5,
        "distance_km": 350.0,
        "age": 45,
        "gender": "F",
        "city_pop": 120000
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["fraud_probability"] > 0.10


def test_prediction_invalid_payload(client):
    """Tests POST /api/predict with negative amount error handling."""
    payload = {
        "amt": -50.0,
        "category": "grocery_pos"
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert "cannot be negative" in data["message"].lower()


def test_metrics_endpoint(client):
    """Tests GET /api/metrics."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "test_metrics" in data
    assert "model_comparison" in data
    assert len(data["model_comparison"]) > 0


def test_model_info_endpoint(client):
    """Tests GET /api/model-info."""
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] in ["success", "partial"]


def test_index_page(client):
    """Tests GET / serves index.html dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Credit Card Fraud Detection" in response.data


def test_health_includes_both_models(client):
    """Verifies that health check reports both supervised and unsupervised models."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert "available_models" in data
    assert "gradient_boosting" in data["available_models"]
    assert "isolation_forest" in data["available_models"]
    assert data["isolation_forest_loaded"] is True


def test_prediction_isolation_forest(client):
    """Tests POST /api/predict explicitly selecting Isolation Forest model."""
    payload = {
        "model": "isolation_forest",
        "amt": 1500.00,
        "category": "shopping_net",
        "hour": 2,
        "day_of_week": 6,
        "distance_km": 280.0,
        "age": 42,
        "gender": "M",
        "city_pop": 80000
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["model_name"] == "Isolation Forest"
    assert data["score_type"] == "anomaly_score"
    assert "anomaly_score" in data
    assert isinstance(data["anomaly_score"], float)
    assert "threshold" in data
    assert data["prediction"] in [0, 1]
    assert data["label"] in ["Fraudulent Transaction", "Legitimate Transaction"]
    assert "score" in data
    assert data["selected_model"] == "Isolation Forest"


def test_prediction_auto_model_selection(client):
    """Verifies that omitted model parameter automatically routes to Gradient Boosting production model."""
    payload = {
        "amt": 150.00,
        "category": "misc_net",
        "hour": 20,
        "day_of_week": 3,
        "distance_km": 15.0,
        "age": 29,
        "gender": "F",
        "city_pop": 75000
    }
    # Notice: 'model' key is omitted completely
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["model_name"] == "Gradient Boosting"
    assert data["selected_model"] == "Gradient Boosting"
    assert data["score_type"] == "fraud_probability"
    assert data["threshold"] == 0.67
    assert "score" in data
    assert data["score"] == data["fraud_probability"]


def test_prediction_backward_compatibility_explicit_gb(client):
    """Verifies backward compatibility when 'model': 'gradient_boosting' is explicitly supplied."""
    payload = {
        "model": "gradient_boosting",
        "amt": 85.00,
        "category": "gas_transport",
        "hour": 11,
        "day_of_week": 1,
        "distance_km": 8.0,
        "age": 52,
        "gender": "M",
        "city_pop": 25000
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["model_name"] == "Gradient Boosting"
    assert data["selected_model"] == "Gradient Boosting"
    assert data["score_type"] == "fraud_probability"


def test_prediction_invalid_model_choice(client):
    """Tests POST /api/predict rejects an unsupported model parameter with HTTP 400."""
    payload = {
        "model": "deep_quantum_network",
        "amt": 50.0,
        "category": "grocery_pos"
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert "unsupported model" in data["message"].lower()


def test_comparison_endpoint(client):
    """Tests GET /api/comparison endpoint returns both models in benchmark results."""
    response = client.get("/api/comparison")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "model_comparison" in data
    assert "isolation_forest_metadata" in data

    models_in_comp = [row["Model"] for row in data["model_comparison"]]
    assert "Isolation Forest" in models_in_comp
    assert "Gradient Boosting" in models_in_comp

    # Ensure no duplicates in comparison table
    assert len(models_in_comp) == len(set(models_in_comp)), "Duplicate models found in comparison table"


def test_isolation_forest_score_orientation():
    """Verifies that higher anomaly scores correspond to more anomalous transactions."""
    predictor = get_predictor()
    assert predictor.is_iso_loaded

    normal_tx = {
        "amt": 18.50,
        "category": "grocery_pos",
        "hour": 14,
        "day_of_week": 2,
        "distance_km": 3.0,
        "age": 36,
        "gender": "F",
        "city_pop": 50000
    }
    outlier_tx = {
        "amt": 4850.00,
        "category": "shopping_net",
        "hour": 3,
        "day_of_week": 6,
        "distance_km": 450.0,
        "age": 75,
        "gender": "M",
        "city_pop": 500
    }

    res_normal = predictor.predict(normal_tx, model_name="isolation_forest")
    res_outlier = predictor.predict(outlier_tx, model_name="isolation_forest")

    # Higher score = more anomalous
    assert res_outlier["anomaly_score"] > res_normal["anomaly_score"], (
        f"Expected outlier score ({res_outlier['anomaly_score']}) to exceed normal score ({res_normal['anomaly_score']})"
    )
