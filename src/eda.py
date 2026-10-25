"""
Exploratory Data Analysis (EDA) Module for Credit Card Fraud Detection
Analyzes dataset characteristics and generates publication-grade visualizations.
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script execution
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import load_data, find_dataset_path, haversine_distance


def set_plotting_style():
    """Applies a clean, modern aesthetic with restrained colors."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 1.2,
        "grid.color": "#F1F5F9",
        "grid.linestyle": "--",
        "figure.titlesize": 14,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def run_eda(sample_size: int = 100000, output_dir: str = "results", static_dir: str = "static/images"):
    """
    Executes full EDA workflow, logs key insights, and saves charts.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)
    set_plotting_style()

    print(f"\n{'='*60}\n  STARTING EXPLORATORY DATA ANALYSIS (EDA)\n{'='*60}")
    df, stats = load_data(sample_size=sample_size)

    # 1. Engineer essential columns for analysis
    print("[EDA] Engineering temporal and geographic features for analysis...")
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365
    
    # Distance
    lats = df["lat"].values
    longs = df["long"].values
    merch_lats = df["merch_lat"].values
    merch_longs = df["merch_long"].values
    df["distance_km"] = [
        haversine_distance(lats[i], longs[i], merch_lats[i], merch_longs[i])
        for i in range(len(lats))
    ]

    total_records = len(df)
    fraud_count = int((df["is_fraud"] == 1).sum())
    legit_count = int((df["is_fraud"] == 0).sum())
    fraud_pct = (fraud_count / total_records) * 100

    print(f"[EDA] Total samples analyzed: {total_records:,}")
    print(f"[EDA] Legitimate transactions: {legit_count:,} ({100 - fraud_pct:.2f}%)")
    print(f"[EDA] Fraudulent transactions: {fraud_count:,} ({fraud_pct:.2f}%)")

    # -------------------------------------------------------------
    # Visualization 1: Class Distribution (Imbalance)
    # -------------------------------------------------------------
    print("[EDA] Generating Class Distribution Chart...")
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = ["#2563EB", "#DC2626"]  # Blue for legit, Crimson Red for fraud
    bars = sns.barplot(
        x=["Legitimate (0)", "Fraudulent (1)"],
        y=[legit_count, fraud_count],
        palette=palette,
        ax=ax,
        edgecolor="#1E293B",
        linewidth=1.2,
    )
    ax.set_title("Credit Card Transaction Class Distribution\n(Severe Imbalance: ~0.58% Fraud)", pad=15, fontweight="bold")
    ax.set_ylabel("Transaction Count", fontweight="bold")
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Annotate bar heights
    for p, count, pct in zip(bars.patches, [legit_count, fraud_count], [100 - fraud_pct, fraud_pct]):
        height = p.get_height()
        ax.annotate(f"{count:,}\n({pct:.2f}%)",
                    (p.get_x() + p.get_width() / 2., height),
                    ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points",
                    fontweight="bold", color="#1E293B", fontsize=10)

    fig.tight_layout()
    chart1_path = os.path.join(output_dir, "eda_class_distribution.png")
    fig.savefig(chart1_path, dpi=300)
    shutil.copy(chart1_path, os.path.join(static_dir, "eda_class_distribution.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # Visualization 2: Transaction Amount Distribution (Log Scale)
    # -------------------------------------------------------------
    print("[EDA] Generating Amount Distribution Chart...")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.kdeplot(
        data=df[df["is_fraud"] == 0]["amt"],
        label=f"Legitimate (Mean: ${df[df['is_fraud'] == 0]['amt'].mean():.2f})",
        color="#2563EB", fill=True, alpha=0.3, log_scale=True, ax=ax
    )
    sns.kdeplot(
        data=df[df["is_fraud"] == 1]["amt"],
        label=f"Fraudulent (Mean: ${df[df['is_fraud'] == 1]['amt'].mean():.2f})",
        color="#DC2626", fill=True, alpha=0.4, log_scale=True, ax=ax
    )
    ax.set_title("Transaction Amount Distribution: Legitimate vs Fraudulent", pad=15, fontweight="bold")
    ax.set_xlabel("Transaction Amount ($ - Log Scale)", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    fig.tight_layout()
    chart2_path = os.path.join(output_dir, "eda_amount_distribution.png")
    fig.savefig(chart2_path, dpi=300)
    shutil.copy(chart2_path, os.path.join(static_dir, "eda_amount_distribution.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # Visualization 3: Fraud Rate by Transaction Hour
    # -------------------------------------------------------------
    print("[EDA] Generating Hourly Fraud Rate Chart...")
    hourly_stats = df.groupby("hour")["is_fraud"].agg(["count", "mean"]).reset_index()
    hourly_stats["fraud_pct"] = hourly_stats["mean"] * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        data=hourly_stats, x="hour", y="fraud_pct",
        marker="o", markersize=8, color="#DC2626", linewidth=2.5, ax=ax
    )
    ax.axvspan(22, 23.9, color="#FEE2E2", alpha=0.7, label="Peak Night Risk (22:00 - 04:00)")
    ax.axvspan(0, 4, color="#FEE2E2", alpha=0.7)
    ax.set_title("Fraud Rate (%) by Hour of Day", pad=15, fontweight="bold")
    ax.set_xlabel("Hour of Day (0 to 23)", fontweight="bold")
    ax.set_ylabel("Fraud Occurrence (%)", fontweight="bold")
    ax.set_xticks(range(0, 24))
    ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    fig.tight_layout()
    chart3_path = os.path.join(output_dir, "eda_fraud_by_hour.png")
    fig.savefig(chart3_path, dpi=300)
    shutil.copy(chart3_path, os.path.join(static_dir, "eda_fraud_by_hour.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # Visualization 4: Fraud Occurrence by Category
    # -------------------------------------------------------------
    print("[EDA] Generating Fraud Rate by Category Chart...")
    cat_stats = df.groupby("category")["is_fraud"].agg(["count", "sum", "mean"]).reset_index()
    cat_stats["fraud_pct"] = cat_stats["mean"] * 100
    cat_stats = cat_stats.sort_values(by="fraud_pct", ascending=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=cat_stats, y="category", x="fraud_pct",
        palette="Reds_r", ax=ax, edgecolor="#1E293B", linewidth=0.8
    )
    ax.set_title("Fraud Occurrence Rate (%) Across Spending Categories", pad=15, fontweight="bold")
    ax.set_xlabel("Fraud Percentage (%)", fontweight="bold")
    ax.set_ylabel("Merchant Category", fontweight="bold")
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f"{width:.2f}%",
                    (width, p.get_y() + p.get_height() / 2.),
                    ha="left", va="center",
                    xytext=(5, 0), textcoords="offset points",
                    color="#1E293B", fontsize=9, fontweight="bold")
    fig.tight_layout()
    chart4_path = os.path.join(output_dir, "eda_fraud_by_category.png")
    fig.savefig(chart4_path, dpi=300)
    shutil.copy(chart4_path, os.path.join(static_dir, "eda_fraud_by_category.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # Visualization 5: Correlation Heatmap for Engineered Numerical Features
    # -------------------------------------------------------------
    print("[EDA] Generating Correlation Heatmap...")
    corr_features = ["is_fraud", "amt", "distance_km", "hour", "day_of_week", "age", "city_pop"]
    corr_matrix = df[corr_features].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".3f", cmap="Blues",
        cbar_kws={"label": "Pearson Correlation Coefficient"},
        ax=ax, linewidths=0.5, linecolor="#CBD5E1", square=True
    )
    ax.set_title("Correlation Heatmap: Fraud & Engineered Numerical Features", pad=15, fontweight="bold")
    fig.tight_layout()
    chart5_path = os.path.join(output_dir, "eda_correlation_matrix.png")
    fig.savefig(chart5_path, dpi=300)
    shutil.copy(chart5_path, os.path.join(static_dir, "eda_correlation_matrix.png"))
    plt.close(fig)

    print(f"\n[EDA Complete] All 5 visualizations successfully generated and saved to '{output_dir}/' and '{static_dir}/'.\n")
    return {
        "total_records": total_records,
        "legit_count": legit_count,
        "fraud_count": fraud_count,
        "fraud_pct": fraud_pct,
        "mean_legit_amt": float(df[df["is_fraud"] == 0]["amt"].mean()),
        "mean_fraud_amt": float(df[df["is_fraud"] == 1]["amt"].mean()),
    }


if __name__ == "__main__":
    run_eda(sample_size=100000)
