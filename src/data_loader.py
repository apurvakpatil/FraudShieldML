"""
Data Loader Module for Credit Card Fraud Detection
Handles robust dataset discovery, streaming inspection, and stratified sampling.
"""

import os
import math
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List


def find_dataset_path(base_dir: Optional[str] = None) -> str:
    """
    Locates the credit card transactions dataset file within the project workspace.
    Prioritizes the uploaded dataset without duplicating or moving it.
    """
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    candidate_paths = [
        os.path.join(base_dir, "credit_card_transactions.csv", "credit_card_transactions.csv"),
        os.path.join(base_dir, "credit_card_transactions.csv"),
        os.path.join(base_dir, "dataset", "credit_card_transactions.csv"),
    ]

    for path in candidate_paths:
        if os.path.isfile(path):
            return os.path.abspath(path)

    # Search recursively for any credit card CSV
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(".csv") and "credit_card" in f.lower():
                return os.path.abspath(os.path.join(root, f))

    raise FileNotFoundError(
        f"Could not locate the credit card transaction dataset in '{base_dir}'. "
        "Please verify that credit_card_transactions.csv is uploaded."
    )


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes great-circle distance in kilometers between two geographic coordinates.
    """
    try:
        lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
        lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
        c = 2 * math.asin(math.sqrt(a))
        return 6371.0 * c  # Earth radius in kilometers
    except Exception:
        return 0.0


def inspect_dataset(dataset_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Performs streaming inspection of the entire dataset to compute exact statistics
    without exhausting memory.
    """
    if dataset_path is None:
        dataset_path = find_dataset_path()

    total_rows = 0
    fraud_count = 0
    legit_count = 0
    missing_counts = None
    column_names = None
    dtypes = None

    chunksize = 100000
    for chunk in pd.read_csv(dataset_path, chunksize=chunksize):
        total_rows += len(chunk)
        if column_names is None:
            column_names = chunk.columns.tolist()
            missing_counts = chunk.isnull().sum()
            dtypes = {col: str(dt) for col, dt in chunk.dtypes.items()}
        else:
            missing_counts += chunk.isnull().sum()

        if "is_fraud" in chunk.columns:
            vc = chunk["is_fraud"].value_counts()
            fraud_count += int(vc.get(1, 0))
            legit_count += int(vc.get(0, 0))

    fraud_percentage = (fraud_count / total_rows * 100) if total_rows > 0 else 0.0

    return {
        "dataset_path": dataset_path,
        "total_rows": total_rows,
        "legit_count": legit_count,
        "fraud_count": fraud_count,
        "fraud_percentage": round(fraud_percentage, 4),
        "target_column": "is_fraud",
        "column_names": column_names,
        "dtypes": dtypes,
        "missing_counts": missing_counts.to_dict() if missing_counts is not None else {},
    }


def load_data(
    sample_size: Optional[int] = 100000,
    random_state: int = 42,
    dataset_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads dataset with optional stratified sampling to ensure exact class distribution
    while keeping model training fast, stable, and reproducible.
    """
    if dataset_path is None:
        dataset_path = find_dataset_path()

    print(f"[DataLoader] Reading dataset from: {dataset_path}")

    # Inspect total row count first if sampling
    if sample_size is not None and sample_size > 0:
        print(f"[DataLoader] Stratified sampling {sample_size:,} records from full dataset...")
        np.random.seed(random_state)
        
        # Read chunks and stratify
        legit_chunks = []
        fraud_chunks = []
        chunksize = 150000

        # We know total ~ 1,296,675 rows, ~0.58% fraud.
        # Desired fraud rows = sample_size * 0.0058 (approx 580)
        # Desired legit rows = sample_size - desired_fraud
        target_fraud = max(100, int(sample_size * 0.005789))
        target_legit = sample_size - target_fraud

        total_fraud_collected = 0
        total_legit_collected = 0

        # Pass 1: Collect all fraud rows (total fraud is ~7506) and sample legit rows proportionally
        legit_pool = []
        fraud_pool = []

        for chunk in pd.read_csv(dataset_path, chunksize=chunksize):
            fraud_subset = chunk[chunk["is_fraud"] == 1]
            legit_subset = chunk[chunk["is_fraud"] == 0]
            
            fraud_pool.append(fraud_subset)
            # Sample legit rows from this chunk
            legit_sample_frac = target_legit / 1290000.0 * 1.05
            sampled_legit = legit_subset.sample(
                frac=min(1.0, legit_sample_frac),
                random_state=random_state
            )
            legit_pool.append(sampled_legit)

        all_frauds = pd.concat(fraud_pool, ignore_index=True)
        all_legits = pd.concat(legit_pool, ignore_index=True)

        # Sample exact target counts
        sampled_frauds = all_frauds.sample(n=min(target_fraud, len(all_frauds)), random_state=random_state)
        sampled_legits = all_legits.sample(n=min(target_legit, len(all_legits)), random_state=random_state)

        df = pd.concat([sampled_frauds, sampled_legits], ignore_index=True)
        df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    else:
        print("[DataLoader] Loading complete dataset into memory...")
        df = pd.read_csv(dataset_path)

    stats = {
        "shape": df.shape,
        "legit_count": int((df["is_fraud"] == 0).sum()),
        "fraud_count": int((df["is_fraud"] == 1).sum()),
        "fraud_percentage": round((df["is_fraud"] == 1).mean() * 100, 4),
    }

    print(
        f"[DataLoader] Loaded {len(df):,} rows. Legit: {stats['legit_count']:,}, "
        f"Fraud: {stats['fraud_count']:,} ({stats['fraud_percentage']}%)"
    )
    return df, stats


def get_sample_transactions(dataset_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Extracts genuine benchmark transactions (both fraudulent and legitimate)
    from the original dataset for demonstration and web UI testing.
    """
    if dataset_path is None:
        dataset_path = find_dataset_path()

    # Read a sample chunk
    chunk = pd.read_csv(dataset_path, nrows=50000)
    
    # Pre-extract distance, age, hour, day_of_week
    lat = chunk["lat"].values
    long = chunk["long"].values
    merch_lat = chunk["merch_lat"].values
    merch_long = chunk["merch_long"].values
    distances = [haversine_distance(lat[i], long[i], merch_lat[i], merch_long[i]) for i in range(len(lat))]
    
    trans_time = pd.to_datetime(chunk["trans_date_trans_time"])
    dob = pd.to_datetime(chunk["dob"])
    ages = ((trans_time - dob).dt.days // 365).values
    hours = trans_time.dt.hour.values
    dows = trans_time.dt.dayofweek.values

    chunk["distance_km"] = [round(d, 2) for d in distances]
    chunk["age"] = ages
    chunk["hour"] = hours
    chunk["day_of_week"] = dows

    samples = []

    # Get 2 legitimate samples
    legit_df = chunk[chunk["is_fraud"] == 0]
    for idx, row in legit_df.iloc[[5, 25]].iterrows():
        samples.append({
            "name": f"Legitimate: Normal {row['category'].replace('_', ' ').title()} (${row['amt']:.2f})",
            "is_fraud": 0,
            "category": str(row["category"]),
            "amt": float(row["amt"]),
            "gender": str(row["gender"]),
            "age": int(row["age"]),
            "city_pop": int(row["city_pop"]),
            "hour": int(row["hour"]),
            "day_of_week": int(row["day_of_week"]),
            "distance_km": float(row["distance_km"]),
            "description": f"Standard legitimate transaction for ${row['amt']:.2f} during daytime (hour {row['hour']}).",
        })

    # Get 2 fraudulent samples
    fraud_df = chunk[chunk["is_fraud"] == 1]
    if len(fraud_df) >= 2:
        for idx, row in fraud_df.iloc[[0, min(1, len(fraud_df) - 1)]].iterrows():
            samples.append({
                "name": f"Fraud Alert: High-Risk {row['category'].replace('_', ' ').title()} (${row['amt']:.2f})",
                "is_fraud": 1,
                "category": str(row["category"]),
                "amt": float(row["amt"]),
                "gender": str(row["gender"]),
                "age": int(row["age"]),
                "city_pop": int(row["city_pop"]),
                "hour": int(row["hour"]),
                "day_of_week": int(row["day_of_week"]),
                "distance_km": float(row["distance_km"]),
                "description": f"Known fraudulent transaction of ${row['amt']:.2f} in {row['category']} at hour {row['hour']}.",
            })

    return samples


if __name__ == "__main__":
    print("[Test] Testing data_loader module...")
    path = find_dataset_path()
    print("Found dataset at:", path)
    df, stats = load_data(sample_size=10000)
    print("Sample stats:", stats)
    sample_txs = get_sample_transactions()
    print(f"Extracted {len(sample_txs)} genuine sample transactions.")
