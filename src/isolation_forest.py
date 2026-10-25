"""
Isolation Forest Training Module

Utility functions to train an IsolationForest model for fraud detection.
The model is trained **only on legitimate transactions** (y == 0) from the
pure training partition.  Validation and test sets are used *solely* for
threshold selection and evaluation.
"""

import os
import sys
import json
import time
import joblib
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
from typing import Tuple, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local imports – reuse existing preprocessing utilities
from src.preprocessing import build_full_preprocessing_pipeline, ALL_MODEL_FEATURES, TARGET_COLUMN
from src.evaluate_model import find_optimal_threshold, calculate_metrics


def train_isolation_forest(
    X_train, y_train, X_val, y_val, X_test, y_test,
    random_state: int = 42,
    contamination: float = 0.01,
    model_dir: str = "models",
) -> Dict[str, Any]:
    """Train IsolationForest on *legitimate* training data.

    Parameters
    ----------
    X_train, y_train : pd.DataFrame, pd.Series
        Full training partition (80 % of the original data).
    X_val, y_val : pd.DataFrame, pd.Series
        Validation partition (20 % of training data).
    X_test, y_test : pd.DataFrame, pd.Series
        Held‑out test partition (20 % of the original data).
    random_state : int, default 42
        Seed for reproducibility.
    contamination : float, default 0.01
        Expected proportion of outliers in the *legitimate* training set.
        This is only an initial setting; the operational threshold is chosen
        on the validation data via ``find_optimal_threshold``.
    model_dir : str, default "models"
        Directory where the model artifact and metadata will be saved.

    Returns
    -------
    dict
        A dictionary containing the trained model, selected threshold, and
        evaluation metrics for the test set.
    """
    # ---------------------------------------------------------------------
    # 1. Build and fit the shared preprocessing pipeline on the pure train set
    # ---------------------------------------------------------------------
    preprocessor = build_full_preprocessing_pipeline()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    # ---------------------------------------------------------------------
    # 2. Sub‑select only legitimate records for IsolationForest fitting
    # ---------------------------------------------------------------------
    legit_mask = (y_train == 0)
    X_legit = X_train_proc[legit_mask]
    # Verify that no fraudulent rows are present – raise if accidental leakage
    if np.any(y_train[legit_mask] != 0):
        raise RuntimeError("IsolationForest training data contains fraud records.")

    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    # Time only the model fitting (excludes preprocessing and scoring)
    t_start = time.perf_counter()
    iso_forest.fit(X_legit)
    training_time = round(time.perf_counter() - t_start, 2)

    # ---------------------------------------------------------------------
    # 3. Compute anomaly scores (higher = more anomalous) for validation & test
    # ---------------------------------------------------------------------
    def anomaly_score(X):
        # decision_function returns higher values for inliers; invert
        return -iso_forest.decision_function(X)

    val_scores = anomaly_score(X_val_proc)
    test_scores = anomaly_score(X_test_proc)

    # ---------------------------------------------------------------------
    # 4. Choose operational threshold on validation data
    # ---------------------------------------------------------------------
    optimal_threshold = find_optimal_threshold(y_val, val_scores, metric="f1")

    # ---------------------------------------------------------------------
    # 5. Classification on test set using the selected threshold
    # ---------------------------------------------------------------------
    y_pred_test = (test_scores >= optimal_threshold).astype(int)
    test_metrics = calculate_metrics(y_test, y_pred_test, test_scores)

    # ---------------------------------------------------------------------
    # 6. Persist model and metadata (do not overwrite existing Gradient Boosting artifact)
    # ---------------------------------------------------------------------
    # Persist both the trained IsolationForest and the preprocessing pipeline.
    # Store as a tuple so that inference can reuse identical transformations.
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "isolation_forest_model.joblib")
    joblib.dump((preprocessor, iso_forest), model_path)
    metadata = {
        "model_name": "Isolation Forest",
        "model_version": "1.0.0",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_column": TARGET_COLUMN,
        "input_features": ALL_MODEL_FEATURES,
        "contamination": contamination,
        "selected_threshold": optimal_threshold,
        "random_seed": random_state,
        "training_time_seconds": training_time,
        "test_metrics": test_metrics,
    }
    meta_path = os.path.join(model_dir, "isolation_forest_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return {
        "model": iso_forest,
        "preprocessor": preprocessor,
        "threshold": optimal_threshold,
        "test_metrics": test_metrics,
        "test_scores": test_scores,
        "training_time": training_time,
        "model_path": model_path,
        "metadata_path": meta_path,
    }

# The module can also be executed directly for quick debugging
if __name__ == "__main__":
    from src.data_loader import load_data, find_dataset_path
    from src.preprocessing import prepare_train_test_data
    dataset_path = find_dataset_path()
    df, _ = load_data(sample_size=20000, dataset_path=dataset_path)
    X_tr, X_te, y_tr, y_te = prepare_train_test_data(df)
    # Create validation split (same logic as main training script)
    from sklearn.model_selection import train_test_split
    X_trn, X_val, y_trn, y_val = train_test_split(
        X_tr, y_tr, test_size=0.25, random_state=42, stratify=y_tr
    )
    result = train_isolation_forest(
        X_trn, y_trn, X_val, y_val, X_te, y_te, random_state=42
    )
    print("Isolation Forest training completed. Test metrics:")
    print(result["test_metrics"])
