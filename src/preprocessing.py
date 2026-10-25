"""
Preprocessing and Feature Engineering Module for Credit Card Fraud Detection
Implements scikit-learn compliant feature engineering, transformations, and leak-free pipelines.
"""

import math
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


# Define consistent feature schema
NUMERICAL_FEATURES = ["amt", "distance_km", "hour", "day_of_week", "age", "city_pop"]
CATEGORICAL_FEATURES = ["category", "gender"]
ALL_MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "is_fraud"

# High-risk categories known from fraud pattern analysis
KNOWN_CATEGORIES = [
    "misc_net", "grocery_pos", "entertainment", "gas_transport",
    "misc_pos", "grocery_net", "shopping_net", "shopping_pos",
    "food_dining", "personal_care", "health_fitness", "travel",
    "kids_pets", "home"
]


def haversine_distance_vectorized(lat1, lon1, lat2, lon2):
    """Vectorized Haversine distance calculation in kilometers."""
    try:
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        return 6371.0 * c
    except Exception:
        return np.zeros(len(lat1))


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transforms raw transaction data or handles pre-engineered features.
    Extracts time-of-day (hour), day-of-week, cardholder age, and geographic distance
    between cardholder and merchant while cleanly removing non-predictive identifiers.
    """

    def __init__(self):
        self.numerical_cols = NUMERICAL_FEATURES
        self.categorical_cols = CATEGORICAL_FEATURES

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert dict or array-like to DataFrame if needed
        if isinstance(X, dict):
            X_df = pd.DataFrame([X])
        elif isinstance(X, list):
            X_df = pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X)

        out_df = pd.DataFrame(index=X_df.index)

        # 1. Transaction Amount
        if "amt" in X_df.columns:
            out_df["amt"] = pd.to_numeric(X_df["amt"], errors="coerce").fillna(0.0)
        else:
            out_df["amt"] = 0.0

        # 2. Distance in Kilometers
        if "distance_km" in X_df.columns and not X_df["distance_km"].isnull().all():
            out_df["distance_km"] = pd.to_numeric(X_df["distance_km"], errors="coerce").fillna(50.0)
        elif all(k in X_df.columns for k in ["lat", "long", "merch_lat", "merch_long"]):
            lats = pd.to_numeric(X_df["lat"], errors="coerce").fillna(0.0).values
            longs = pd.to_numeric(X_df["long"], errors="coerce").fillna(0.0).values
            merch_lats = pd.to_numeric(X_df["merch_lat"], errors="coerce").fillna(0.0).values
            merch_longs = pd.to_numeric(X_df["merch_long"], errors="coerce").fillna(0.0).values
            dist = haversine_distance_vectorized(lats, longs, merch_lats, merch_longs)
            out_df["distance_km"] = np.nan_to_num(dist, nan=50.0)
        else:
            out_df["distance_km"] = 50.0

        # 3. Hour and Day of Week
        if "hour" in X_df.columns and not X_df["hour"].isnull().all():
            out_df["hour"] = pd.to_numeric(X_df["hour"], errors="coerce").fillna(12).astype(int)
        elif "trans_date_trans_time" in X_df.columns:
            dt = pd.to_datetime(X_df["trans_date_trans_time"], errors="coerce")
            out_df["hour"] = dt.dt.hour.fillna(12).astype(int)
        else:
            out_df["hour"] = 12

        if "day_of_week" in X_df.columns and not X_df["day_of_week"].isnull().all():
            out_df["day_of_week"] = pd.to_numeric(X_df["day_of_week"], errors="coerce").fillna(2).astype(int)
        elif "trans_date_trans_time" in X_df.columns:
            dt = pd.to_datetime(X_df["trans_date_trans_time"], errors="coerce")
            out_df["day_of_week"] = dt.dt.dayofweek.fillna(2).astype(int)
        else:
            out_df["day_of_week"] = 2

        # 4. Age
        if "age" in X_df.columns and not X_df["age"].isnull().all():
            out_df["age"] = pd.to_numeric(X_df["age"], errors="coerce").fillna(45).astype(int)
        elif "dob" in X_df.columns and "trans_date_trans_time" in X_df.columns:
            dob = pd.to_datetime(X_df["dob"], errors="coerce")
            dt = pd.to_datetime(X_df["trans_date_trans_time"], errors="coerce")
            age_series = (dt - dob).dt.days // 365
            out_df["age"] = age_series.fillna(45).astype(int)
        else:
            out_df["age"] = 45

        # 5. City Population
        if "city_pop" in X_df.columns:
            out_df["city_pop"] = pd.to_numeric(X_df["city_pop"], errors="coerce").fillna(50000).astype(float)
        else:
            out_df["city_pop"] = 50000.0

        # 6. Category
        if "category" in X_df.columns:
            out_df["category"] = X_df["category"].astype(str).fillna("misc_pos")
        else:
            out_df["category"] = "misc_pos"

        # 7. Gender
        if "gender" in X_df.columns:
            out_df["gender"] = X_df["gender"].astype(str).str.upper().fillna("M")
        else:
            out_df["gender"] = "M"

        # Return DataFrame with consistent column ordering
        return out_df[ALL_MODEL_FEATURES]


def get_column_preprocessor() -> ColumnTransformer:
    """
    Builds the ColumnTransformer for scaling numerical features and one-hot encoding categories.
    Uses RobustScaler to handle heavy financial transaction outliers.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )
    return preprocessor


def build_full_preprocessing_pipeline() -> Pipeline:
    """
    Constructs the complete end-to-end preprocessing pipeline.
    Combines feature engineering + column transformations.
    """
    return Pipeline([
        ("engineer", FeatureEngineer()),
        ("transform", get_column_preprocessor()),
    ])


def prepare_train_test_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Separates features and target, and creates a leak-free stratified train-test split.
    Guarantees that test data is never touched during training or threshold selection.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataframe columns: {df.columns.tolist()}")

    y = df[TARGET_COLUMN].astype(int)
    X = df.drop(columns=[TARGET_COLUMN])

    # Stratified split to strictly preserve the imbalanced fraud ratio in both train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"[Preprocessing] Train set: {len(X_train):,} samples (Frauds: {int(y_train.sum()):,}, {y_train.mean()*100:.3f}%)")
    print(f"[Preprocessing] Test set:  {len(X_test):,} samples (Frauds: {int(y_test.sum()):,}, {y_test.mean()*100:.3f}%)")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from src.data_loader import load_data
    df, _ = load_data(sample_size=5000)
    X_train, X_test, y_train, y_test = prepare_train_test_data(df)
    pipe = build_full_preprocessing_pipeline()
    X_train_proc = pipe.fit_transform(X_train)
    print("Transformed train shape:", X_train_proc.shape)
