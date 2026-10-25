"""
Inference and Prediction Service for Credit Card Fraud Detection
Loads the trained pipeline artifact, validates inputs, and generates predictions with risk interpretations.
"""

import os
import sys
import json
import joblib
import pandas as pd
from typing import Dict, Any, Union, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import (
    ALL_MODEL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    KNOWN_CATEGORIES,
)


class FraudPredictor:
    """
    Singleton wrapper for loading model artifact, validating requests,
    and producing risk-scored transaction classifications.
    """

    def __init__(self, model_path: Optional[str] = None, metadata_path: Optional[str] = None):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if model_path is None:
            model_path = os.path.join(base_dir, "models", "fraud_detection_model.joblib")
        if metadata_path is None:
            metadata_path = os.path.join(base_dir, "models", "model_metadata.json")

        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata = {}
        self.threshold = 0.50
        self.model_name = "Ensemble Model"
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"[FraudPredictor] Warning: Model file not found at {self.model_path}. Train model first.")
            return

        print(f"[FraudPredictor] Loading model from: {self.model_path}")
        self.model = joblib.load(self.model_path)

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
                self.threshold = self.metadata.get("selected_threshold", 0.50)
                self.model_name = self.metadata.get("model_name", "Ensemble Model")
        print(f"[FraudPredictor] Successfully loaded '{self.model_name}' (threshold: {self.threshold})")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def validate_features(self, data: Dict[str, Any]) -> tuple:
        """
        Validates incoming transaction features against expected types and ranges.
        Returns (is_valid, cleaned_dict, error_message).
        """
        if not isinstance(data, dict):
            return False, {}, "Input payload must be a JSON object containing transaction features."

        cleaned = {}

        # 1. Amount
        if "amt" not in data:
            return False, {}, "Missing required field: 'amt' (Transaction Amount in USD)."
        try:
            amt = float(data["amt"])
            if amt < 0:
                return False, {}, "Transaction amount 'amt' cannot be negative."
            cleaned["amt"] = amt
        except (ValueError, TypeError):
            return False, {}, f"Invalid numerical value for 'amt': {data.get('amt')}"

        # 2. Category
        cat = str(data.get("category", "misc_pos")).strip().lower()
        if cat not in KNOWN_CATEGORIES:
            # If not in exact list, map to closest or accept as is (the OneHotEncoder ignores unknown gracefully)
            cat = cat if cat else "misc_pos"
        cleaned["category"] = cat

        # 3. Gender
        gender = str(data.get("gender", "M")).strip().upper()
        if gender not in ["M", "F"]:
            gender = "M"
        cleaned["gender"] = gender

        # 4. Numerical features with safe bounds
        try:
            cleaned["age"] = max(18, min(100, int(data.get("age", 40))))
        except Exception:
            cleaned["age"] = 40

        try:
            cleaned["city_pop"] = max(100, float(data.get("city_pop", 50000)))
        except Exception:
            cleaned["city_pop"] = 50000.0

        try:
            cleaned["hour"] = max(0, min(23, int(data.get("hour", 12))))
        except Exception:
            cleaned["hour"] = 12

        try:
            cleaned["day_of_week"] = max(0, min(6, int(data.get("day_of_week", 2))))
        except Exception:
            cleaned["day_of_week"] = 2

        try:
            cleaned["distance_km"] = max(0.0, float(data.get("distance_km", 25.0)))
        except Exception:
            cleaned["distance_km"] = 25.0

        return True, cleaned, ""

    def predict(self, feature_data: Dict[str, Any], custom_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Executes prediction on input features dictionary.
        Returns comprehensive dictionary with class, label, probability, risk level, and explanation.
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Please train the model using `python -m src.train_model` first.")

        valid, cleaned_data, err_msg = self.validate_features(feature_data)
        if not valid:
            raise ValueError(err_msg)

        threshold = custom_threshold if custom_threshold is not None else self.threshold

        # Format input as a 1-row DataFrame
        df_input = pd.DataFrame([cleaned_data])

        # Get probability
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(df_input)[0]
            fraud_prob = float(probs[1])
        elif hasattr(self.model, "decision_function"):
            dfunc = float(self.model.decision_function(df_input)[0])
            # Sigmoid approximation
            fraud_prob = float(1.0 / (1.0 + (2.71828 ** (-dfunc))))
        else:
            pred = int(self.model.predict(df_input)[0])
            fraud_prob = 1.0 if pred == 1 else 0.0

        is_fraud = int(fraud_prob >= threshold)

        # Risk level categorization
        if fraud_prob >= 0.75:
            risk_level = "Critical Risk"
        elif fraud_prob >= threshold:
            risk_level = "High Risk"
        elif fraud_prob >= 0.20:
            risk_level = "Moderate Attention"
        else:
            risk_level = "Low Risk"

        label = "Fraudulent Transaction" if is_fraud == 1 else "Legitimate Transaction"

        # Generate rule/domain-based explanatory indicators
        risk_factors = []
        if cleaned_data["amt"] > 400.0:
            risk_factors.append(f"High transaction amount (${cleaned_data['amt']:.2f})")
        if cleaned_data["hour"] in [22, 23, 0, 1, 2, 3]:
            risk_factors.append(f"Off-peak night hour ({cleaned_data['hour']}:00)")
        if cleaned_data["category"] in ["shopping_net", "grocery_pos", "misc_net"]:
            risk_factors.append(f"Elevated risk merchant category '{cleaned_data['category']}'")
        if cleaned_data["distance_km"] > 150.0:
            risk_factors.append(f"Unusual distance from cardholder location ({cleaned_data['distance_km']:.1f} km)")

        if is_fraud == 1:
            explanation = "The ensemble model classified this transaction as POTENTIALLY FRAUDULENT. "
            if risk_factors:
                explanation += "Key contributing factors: " + ", ".join(risk_factors) + "."
            else:
                explanation += "Statistical pattern aligns closely with historical fraudulent behaviors."
        else:
            explanation = "The ensemble model classified this transaction as LIKELY LEGITIMATE. "
            if risk_factors:
                explanation += f"Note: Moderate flags detected ({', '.join(risk_factors)}), but overall score remains below decision threshold."
            else:
                explanation += "Transaction attributes match standard legitimate consumer purchasing behavior."

        return {
            "prediction": is_fraud,
            "label": label,
            "fraud_probability": round(fraud_prob, 4),
            "legit_probability": round(1.0 - fraud_prob, 4),
            "threshold": round(threshold, 2),
            "risk_level": risk_level,
            "model_name": self.model_name,
            "explanation": explanation,
            "risk_factors": risk_factors,
            "cleaned_features": cleaned_data,
            "disclaimer": "This prediction is generated by an ensemble machine learning model for risk screening and decision support. It is not an absolute financial guarantee.",
        }


Tuple_Validation = tuple[bool, Dict[str, Any], str]

# Global instance for Flask application
predictor_instance: Optional[FraudPredictor] = None

def get_predictor() -> FraudPredictor:
    global predictor_instance
    if predictor_instance is None:
        predictor_instance = FraudPredictor()
    return predictor_instance
