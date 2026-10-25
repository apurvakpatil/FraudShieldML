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


def test_prediction_legitimate(client):
    """Tests POST /api/predict with standard low-risk transaction."""
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
