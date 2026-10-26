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
    MAX_REALISTIC_AMOUNT = 30000.0
    MAX_REALISTIC_DISTANCE_KM = 20015.0

    def __init__(
        self,
        model_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        iso_model_path: Optional[str] = None,
        iso_metadata_path: Optional[str] = None,
    ):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if model_path is None:
            model_path = os.path.join(base_dir, "models", "fraud_detection_model.joblib")
        if metadata_path is None:
            metadata_path = os.path.join(base_dir, "models", "model_metadata.json")
        if iso_model_path is None:
            iso_model_path = os.path.join(base_dir, "models", "isolation_forest_model.joblib")
        if iso_metadata_path is None:
            iso_metadata_path = os.path.join(base_dir, "models", "isolation_forest_metadata.json")

        self.model_path = model_path
        self.metadata_path = metadata_path
        self.iso_model_path = iso_model_path
        self.iso_metadata_path = iso_metadata_path

        # Supervised Gradient Boosting attributes
        self.model = None
        self.metadata = {}
        self.threshold = 0.50
        self.model_name = "Gradient Boosting"
        self.production_model = "Gradient Boosting"
        self.production_model_key = "gradient_boosting"

        # Unsupervised Isolation Forest attributes
        self.iso_model = None
        self.iso_preprocessor = None
        self.iso_metadata = {}
        self.iso_threshold = 0.0
        self.iso_name = "Isolation Forest"

        self._load_model()

    def _load_model(self):
        # 1. Load Supervised Gradient Boosting Model
        if os.path.exists(self.model_path):
            try:
                print(f"[FraudPredictor] Loading supervised model from: {self.model_path}")
                self.model = joblib.load(self.model_path)
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, "r") as f:
                        self.metadata = json.load(f)
                        self.threshold = self.metadata.get("selected_threshold", 0.50)
                        self.model_name = self.metadata.get("model_name", "Gradient Boosting")
                        self.production_model = self.metadata.get("production_model", self.model_name)
                        self.production_model_key = self.metadata.get("production_model_key", "gradient_boosting")
                print(f"[FraudPredictor] Successfully loaded '{self.model_name}' (threshold: {self.threshold}, production: '{self.production_model}')")
            except Exception as e:
                print(f"[FraudPredictor] Warning: Failed to load supervised model: {e}")
                self.model = None
        else:
            print(f"[FraudPredictor] Notice: Supervised model file not found at {self.model_path}.")

        # 2. Load Unsupervised Isolation Forest Model
        if os.path.exists(self.iso_model_path):
            try:
                print(f"[FraudPredictor] Loading Isolation Forest from: {self.iso_model_path}")
                iso_artifact = joblib.load(self.iso_model_path)
                if isinstance(iso_artifact, tuple) and len(iso_artifact) == 2:
                    self.iso_preprocessor, self.iso_model = iso_artifact
                else:
                    self.iso_model = iso_artifact
                if os.path.exists(self.iso_metadata_path):
                    with open(self.iso_metadata_path, "r") as f:
                        self.iso_metadata = json.load(f)
                        self.iso_threshold = self.iso_metadata.get("selected_threshold", 0.0)
                        self.iso_name = self.iso_metadata.get("model_name", "Isolation Forest")
                print(f"[FraudPredictor] Successfully loaded '{self.iso_name}' (threshold: {self.iso_threshold})")
            except Exception as e:
                print(f"[FraudPredictor] Warning: Failed to load Isolation Forest model: {e}")
                self.iso_model = None
                self.iso_preprocessor = None
        else:
            print(f"[FraudPredictor] Notice: Isolation Forest model file not found at {self.iso_model_path}.")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def is_iso_loaded(self) -> bool:
        return self.iso_model is not None and self.iso_preprocessor is not None

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
            if amt > self.MAX_REALISTIC_AMOUNT:
                return False, {}, (
                    f"Transaction amount ${amt:,.2f} exceeds the realistic maximum "
                    f"(${self.MAX_REALISTIC_AMOUNT:,.2f}) observed in training data. "
                    "This value is outside the model's valid operating range. "
                    "Please verify the entered amount."
                )
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
            distance_km = max(0.0, float(data.get("distance_km", 25.0)))
            if distance_km > self.MAX_REALISTIC_DISTANCE_KM:
                return False, {}, (
                    f"Distance {distance_km:,.1f} km exceeds Earth's maximum possible "
                    f"surface distance (~{self.MAX_REALISTIC_DISTANCE_KM:,.0f} km). "
                    "Please verify the entered location data."
                )
            cleaned["distance_km"] = distance_km
        except Exception:
            cleaned["distance_km"] = 25.0

        return True, cleaned, ""

    def _extract_risk_factors(self, cleaned_data: Dict[str, Any]) -> List[str]:
        """Generates domain-rule risk indicators based on transaction features."""
        risk_factors = []
        if cleaned_data["amt"] > 400.0:
            risk_factors.append(f"High transaction amount (${cleaned_data['amt']:.2f})")
        if cleaned_data["hour"] in [22, 23, 0, 1, 2, 3]:
            risk_factors.append(f"Off-peak night hour ({cleaned_data['hour']}:00)")
        if cleaned_data["category"] in ["shopping_net", "grocery_pos", "misc_net"]:
            risk_factors.append(f"Elevated risk merchant category '{cleaned_data['category']}'")
        if cleaned_data["distance_km"] > 150.0:
            risk_factors.append(f"Unusual distance from cardholder location ({cleaned_data['distance_km']:.1f} km)")
        return risk_factors

    def _predict_supervised(self, cleaned_data: Dict[str, Any], custom_threshold: Optional[float] = None) -> Dict[str, Any]:
        """Executes inference using the supervised Gradient Boosting pipeline."""
        if not self.is_loaded:
            raise RuntimeError("Fraud detection model is not loaded. Please train the model using `python -m src.train_model` first.")

        threshold = custom_threshold if custom_threshold is not None else self.threshold
        df_input = pd.DataFrame([cleaned_data])

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(df_input)[0]
            fraud_prob = float(probs[1])
        elif hasattr(self.model, "decision_function"):
            dfunc = float(self.model.decision_function(df_input)[0])
            fraud_prob = float(1.0 / (1.0 + (2.71828 ** (-dfunc))))
        else:
            pred = int(self.model.predict(df_input)[0])
            fraud_prob = 1.0 if pred == 1 else 0.0

        is_fraud = int(fraud_prob >= threshold)

        if fraud_prob >= 0.75:
            risk_level = "Critical Risk"
        elif fraud_prob >= threshold:
            risk_level = "High Risk"
        elif fraud_prob >= 0.20:
            risk_level = "Moderate Attention"
        else:
            risk_level = "Low Risk"

        label = "Fraudulent Transaction" if is_fraud == 1 else "Legitimate Transaction"
        risk_factors = self._extract_risk_factors(cleaned_data)

        # Hybrid rule-based escalation for near-boundary high-risk values
        # that ML alone may underweight due to limited training exposure
        if len(risk_factors) >= 3 and risk_level == "Low Risk":
            risk_level = "Moderate Attention"
        elif len(risk_factors) >= 3 and risk_level == "Moderate Attention":
            risk_level = "High Risk"

        if is_fraud == 1:
            explanation = "The supervised ensemble model classified this transaction as POTENTIALLY FRAUDULENT. "
            if risk_factors:
                explanation += "Key contributing factors: " + ", ".join(risk_factors) + "."
            else:
                explanation += "Statistical pattern aligns closely with historical fraudulent behaviors."
        else:
            explanation = "The supervised ensemble model classified this transaction as LIKELY LEGITIMATE. "
            if risk_factors:
                explanation += f"Note: Moderate flags detected ({', '.join(risk_factors)}), but overall score remains below decision threshold."
            else:
                explanation += "Transaction attributes match standard legitimate consumer purchasing behavior."


        return {
            "prediction": is_fraud,
            "label": label,
            "score_type": "fraud_probability",
            "score": round(fraud_prob, 4),
            "fraud_probability": round(fraud_prob, 4),
            "legit_probability": round(1.0 - fraud_prob, 4),
            "anomaly_score": None,
            "threshold": round(threshold, 2),
            "risk_level": risk_level,
            "model_name": self.model_name,
            "selected_model": self.model_name,
            "explanation": explanation,
            "risk_factors": risk_factors,
            "cleaned_features": cleaned_data,
            "disclaimer": "This prediction is generated by an ensemble machine learning model for risk screening and decision support. It is not an absolute financial guarantee.",
        }

    def _predict_isolation_forest(self, cleaned_data: Dict[str, Any], custom_threshold: Optional[float] = None) -> Dict[str, Any]:
        """Executes inference using the unsupervised Isolation Forest anomaly detection model."""
        if not self.is_iso_loaded:
            raise RuntimeError("Isolation Forest model is not loaded. Train the model first.")

        threshold = custom_threshold if custom_threshold is not None else self.iso_threshold
        df_input = pd.DataFrame([cleaned_data])
        X_proc = self.iso_preprocessor.transform(df_input)

        # Higher anomaly score indicates greater outlierness
        anomaly_score = float(-self.iso_model.decision_function(X_proc)[0])
        is_fraud = int(anomaly_score >= threshold)

        if anomaly_score >= threshold + 0.05:
            risk_level = "Critical Risk"
        elif anomaly_score >= threshold:
            risk_level = "High Risk"
        elif anomaly_score >= threshold - 0.05:
            risk_level = "Moderate Attention"
        else:
            risk_level = "Low Risk"

        label = "Fraudulent Transaction" if is_fraud == 1 else "Legitimate Transaction"
        risk_factors = self._extract_risk_factors(cleaned_data)

        if is_fraud == 1:
            explanation = (
                f"The unsupervised Isolation Forest model flagged this transaction as ANOMALOUS (Score: {anomaly_score:.4f} >= Threshold: {threshold:.4f}). "
                "The transaction exhibits feature isolation characteristics atypical of legitimate behavior."
            )
            if risk_factors:
                explanation += " Notable risk flags: " + ", ".join(risk_factors) + "."
        else:
            explanation = (
                f"The unsupervised Isolation Forest model classified this transaction as NORMAL / INLIER (Score: {anomaly_score:.4f} < Threshold: {threshold:.4f}). "
                "Feature combinations align with typical transaction patterns in feature space."
            )
            if risk_factors:
                explanation += f" Note: Minor flags observed ({', '.join(risk_factors)}), but overall score remains within inlier limits."

        return {
            "prediction": is_fraud,
            "label": label,
            "score_type": "anomaly_score",
            "score": round(anomaly_score, 4),
            "anomaly_score": round(anomaly_score, 4),
            "fraud_probability": None,
            "legit_probability": None,
            "threshold": round(threshold, 4),
            "risk_level": risk_level,
            "model_name": self.iso_name,
            "selected_model": self.iso_name,
            "explanation": explanation,
            "risk_factors": risk_factors,
            "cleaned_features": cleaned_data,
            "disclaimer": "This prediction is generated by an unsupervised Isolation Forest model for outlier risk screening. It is not an absolute financial guarantee.",
        }

    def predict(
        self,
        feature_data: Dict[str, Any],
        custom_threshold: Optional[float] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes prediction on input features dictionary.
        Automatically uses the configured production model (Gradient Boosting) when model_name is omitted or None.
        Supports explicit model_name ('gradient_boosting' or 'isolation_forest') for testing and comparison.
        """
        valid, cleaned_data, err_msg = self.validate_features(feature_data)
        if not valid:
            raise ValueError(err_msg)

        # Automatic model selection if not explicitly specified
        if model_name is None or (isinstance(model_name, str) and not model_name.strip()):
            target_model = self.production_model_key
        else:
            norm = str(model_name).strip().lower().replace("-", "_").replace(" ", "_")
            if norm in ["isolation_forest", "isolationforest", "iso_forest", "iforest", "unsupervised"]:
                target_model = "isolation_forest"
            elif norm in ["gradient_boosting", "gradientboosting", "gb", "supervised", "ensemble", "ensemble_model"]:
                target_model = "gradient_boosting"
            else:
                raise ValueError(f"Unsupported model '{model_name}'. Choose from 'gradient_boosting' or 'isolation_forest'.")

        if target_model == "isolation_forest":
            if not self.is_iso_loaded:
                raise RuntimeError("Isolation Forest model is not loaded. Train the model first.")
            return self._predict_isolation_forest(cleaned_data, custom_threshold=custom_threshold)
        else:
            if not self.is_loaded:
                raise RuntimeError("Fraud detection model is not loaded. Please train the model using `python -m src.train_model` first.")
            return self._predict_supervised(cleaned_data, custom_threshold=custom_threshold)


Tuple_Validation = tuple[bool, Dict[str, Any], str]

# Global instance for Flask application
predictor_instance: Optional[FraudPredictor] = None

def get_predictor() -> FraudPredictor:
    global predictor_instance
    if predictor_instance is None:
        predictor_instance = FraudPredictor()
    return predictor_instance
