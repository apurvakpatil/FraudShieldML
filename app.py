"""
Credit Card Fraud Detection Flask Backend Application
Serves RESTful APIs for predictions, model evaluation metrics, model information,
genuine transaction samples, and the interactive web frontend dashboard.
"""

import os
import json
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory

from src.predict import get_predictor
from src.data_loader import get_sample_transactions

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static")
)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")


@app.route("/")
def index():
    """Serves the primary web dashboard."""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Health check endpoint returning application state and model availability.
    """
    predictor = get_predictor()
    return jsonify({
        "status": "ok",
        "model_loaded": predictor.is_loaded,
        "model_name": predictor.model_name,
        "production_model": predictor.production_model,
        "model_selection_strategy": predictor.metadata.get("model_selection_strategy", "validation_f1_and_prauc_optimization"),
        "threshold": predictor.threshold,
        "isolation_forest_loaded": predictor.is_iso_loaded,
        "isolation_forest_threshold": predictor.iso_threshold,
        "available_models": ["gradient_boosting", "isolation_forest"],
        "version": "1.0.0"
    }), 200


@app.route("/api/predict", methods=["POST"])
def predict_transaction():
    """
    Fraud Prediction Endpoint.
    Validates payload, automatically selects the production model internally,
    executes pipeline inference, and returns risk assessment.
    Optional model parameter ('gradient_boosting' or 'isolation_forest') supported for backward compatibility.
    """
    try:
        payload = request.get_json(force=True, silent=False)
        if not payload:
            return jsonify({
                "status": "error",
                "message": "Invalid JSON or empty request payload."
            }), 400

        # Support both {"features": {...}, "model": "..."} and flat payload {...}
        model_choice = None
        if "features" in payload and isinstance(payload["features"], dict):
            features = payload["features"]
            model_choice = payload.get("model", payload.get("model_name", None))
        else:
            model_choice = payload.get("model", payload.get("model_name", None))
            features = {k: v for k, v in payload.items() if k not in ["model", "model_name"]}

        # If model is explicitly provided, validate it; otherwise use automatic internal selection
        target_model = None
        if model_choice is not None and str(model_choice).strip() != "":
            norm_model = str(model_choice).strip().lower().replace("-", "_").replace(" ", "_")
            if norm_model in ["isolation_forest", "isolationforest", "iso_forest", "iforest", "unsupervised"]:
                target_model = "isolation_forest"
            elif norm_model in ["gradient_boosting", "gradientboosting", "gb", "supervised", "ensemble", "ensemble_model"]:
                target_model = "gradient_boosting"
            else:
                return jsonify({
                    "status": "error",
                    "message": f"Unsupported model '{model_choice}'. Choose from 'gradient_boosting' or 'isolation_forest'."
                }), 400

        predictor = get_predictor()

        # Check required model readiness before executing inference
        effective_model = target_model if target_model is not None else predictor.production_model_key
        if effective_model == "isolation_forest" and not predictor.is_iso_loaded:
            return jsonify({
                "status": "error",
                "message": "Isolation Forest model is not loaded. Train the model first."
            }), 503
        elif effective_model == "gradient_boosting" and not predictor.is_loaded:
            return jsonify({
                "status": "error",
                "message": "Fraud detection model is not loaded. Train the model first."
            }), 503

        # Execute prediction with automatic selection (or optional explicit override)
        result = predictor.predict(features, model_name=target_model)
        return jsonify({
            "status": "success",
            **result
        }), 200

    except ValueError as ve:
        return jsonify({
            "status": "error",
            "message": str(ve)
        }), 400
    except RuntimeError as re:
        return jsonify({
            "status": "error",
            "message": str(re)
        }), 503
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Prediction failed due to an internal error: {str(e)}"
        }), 500


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """
    Returns actual saved model comparison records and test evaluation metrics.
    """
    metrics_path = os.path.join(RESULTS_DIR, "evaluation_metrics.json")
    comparison_path = os.path.join(RESULTS_DIR, "model_comparison.csv")

    metrics_data = {}
    comparison_data = []

    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)

    if os.path.exists(comparison_path):
        df_comp = pd.read_csv(comparison_path)
        comparison_data = df_comp.to_dict(orient="records")

    return jsonify({
        "status": "success",
        "test_metrics": metrics_data,
        "model_comparison": comparison_data
    }), 200


@app.route("/api/comparison", methods=["GET"])
def get_comparison():
    """
    Returns benchmark comparison data across all candidate models including Isolation Forest.
    """
    comparison_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    metrics_path = os.path.join(RESULTS_DIR, "evaluation_metrics.json")
    iso_meta_path = os.path.join(MODELS_DIR, "isolation_forest_metadata.json")

    comparison_data = []
    if os.path.exists(comparison_path):
        df_comp = pd.read_csv(comparison_path)
        comparison_data = df_comp.to_dict(orient="records")

    metrics_data = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)

    iso_meta = {}
    if os.path.exists(iso_meta_path):
        with open(iso_meta_path, "r") as f:
            iso_meta = json.load(f)

    return jsonify({
        "status": "success",
        "model_comparison": comparison_data,
        "supervised_test_metrics": metrics_data,
        "isolation_forest_metadata": iso_meta
    }), 200


@app.route("/api/model-info", methods=["GET"])
def get_model_info():
    """
    Returns architecture, input feature schema, decision threshold, and training metadata.
    """
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    iso_meta_path = os.path.join(MODELS_DIR, "isolation_forest_metadata.json")

    supervised_metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            supervised_metadata = json.load(f)

    iso_metadata = {}
    if os.path.exists(iso_meta_path):
        with open(iso_meta_path, "r") as f:
            iso_metadata = json.load(f)

    predictor = get_predictor()
    return jsonify({
        "status": "success",
        "available_models": ["gradient_boosting", "isolation_forest"],
        "metadata": supervised_metadata,
        "isolation_forest_metadata": iso_metadata,
        "model_loaded": predictor.is_loaded,
        "isolation_forest_loaded": predictor.is_iso_loaded,
    }), 200


@app.route("/api/sample-transactions", methods=["GET"])
def get_samples():
    """
    Provides genuine sample transactions (both legitimate and fraudulent)
    extracted directly from the uploaded dataset for live demonstrations.
    """
    try:
        samples = get_sample_transactions()
        return jsonify({
            "status": "success",
            "samples": samples
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Could not load samples: {str(e)}"
        }), 500


@app.route("/results/<path:filename>")
def serve_results(filename):
    """Serves generated evaluation artifacts and charts directly."""
    return send_from_directory(RESULTS_DIR, filename)


if __name__ == "__main__":
    predictor = get_predictor()
    print(f"[Flask] Server starting. Model loaded: {predictor.is_loaded} ({predictor.model_name})")
    app.run(host="127.0.0.1", port=5000, debug=True)
