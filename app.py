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
        "threshold": predictor.threshold,
        "version": "1.0.0"
    }), 200


@app.route("/api/predict", methods=["POST"])
def predict_transaction():
    """
    Fraud Prediction Endpoint.
    Validates payload, executes pipeline inference, and returns risk assessment.
    """
    try:
        payload = request.get_json(force=True, silent=False)
        if not payload:
            return jsonify({
                "status": "error",
                "message": "Invalid JSON or empty request payload."
            }), 400

        # Support both {"features": {...}} and flat payload {...}
        features = payload.get("features", payload)

        predictor = get_predictor()
        if not predictor.is_loaded:
            return jsonify({
                "status": "error",
                "message": "Fraud detection model is not loaded. Train the model first."
            }), 503

        # Execute prediction
        result = predictor.predict(features)
        return jsonify({
            "status": "success",
            **result
        }), 200

    except ValueError as ve:
        return jsonify({
            "status": "error",
            "message": str(ve)
        }), 400
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


@app.route("/api/model-info", methods=["GET"])
def get_model_info():
    """
    Returns architecture, input feature schema, decision threshold, and training metadata.
    """
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        return jsonify({
            "status": "success",
            "metadata": metadata
        }), 200
    else:
        predictor = get_predictor()
        return jsonify({
            "status": "partial",
            "model_name": predictor.model_name,
            "threshold": predictor.threshold,
            "model_loaded": predictor.is_loaded
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
