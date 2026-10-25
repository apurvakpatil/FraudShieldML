"""
Model Evaluation and Visualization Module for Credit Card Fraud Detection
Computes imbalanced classification metrics, optimizes decision thresholds on validation data,
and generates diagnostic plots (Confusion Matrix, ROC Curve, PR Curve, Feature Importance).
"""

import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def calculate_metrics(y_true, y_pred, y_prob=None) -> dict:
    """
    Computes comprehensive evaluation metrics tailored to imbalanced fraud classification.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    roc_auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0
    pr_auc = average_precision_score(y_true, y_prob) if y_prob is not None else 0.0

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def find_optimal_threshold(y_true, y_prob, metric="f1") -> float:
    """
    Finds the optimal decision threshold on validation data to balance precision and recall.
    Default metric is F1-score. Never evaluated on the test set to avoid leakage.
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    best_score = -1.0
    best_threshold = 0.5

    for th in thresholds:
        preds = (y_prob >= th).astype(int)
        if metric == "f1":
            score = f1_score(y_true, preds, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, preds, zero_division=0)
        else:
            score = f1_score(y_true, preds, zero_division=0)

        if score > best_score:
            best_score = score
            best_threshold = th

    return round(float(best_threshold), 2)


def plot_confusion_matrix(y_true, y_pred, model_name: str, output_path: str, static_path: str = None):
    """Generates and saves an annotated confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues",
        xticklabels=["Legitimate (0)", "Fraudulent (1)"],
        yticklabels=["Legitimate (0)", "Fraudulent (1)"],
        cbar=False, ax=ax, linewidths=1.0, linecolor="#CBD5E1",
        annot_kws={"size": 13, "weight": "bold"}
    )
    ax.set_title(f"Confusion Matrix: {model_name}\n(Evaluated on Untouched Test Set)", pad=15, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Ground-Truth Label", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    if static_path:
        shutil.copy(output_path, static_path)
    plt.close(fig)


def plot_roc_curve(models_dict: dict, y_true, output_path: str, static_path: str = None):
    """Plots and compares ROC curves for multiple models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    palette = ["#2563EB", "#059669", "#D97706", "#7C3AED", "#DC2626", "#0891B2", "#4F46E5"]
    
    idx = 0
    for name, y_prob in models_dict.items():
        if y_prob is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        color = palette[idx % len(palette)]
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=color, linewidth=2.0)
        idx += 1

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.500)")
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", pad=15, fontweight="bold")
    ax.set_xlabel("False Positive Rate (FPR)", fontweight="bold")
    ax.set_ylabel("True Positive Rate / Recall (TPR)", fontweight="bold")
    ax.legend(loc="lower right", frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    if static_path:
        shutil.copy(output_path, static_path)
    plt.close(fig)


def plot_precision_recall_curve(models_dict: dict, y_true, output_path: str, static_path: str = None):
    """
    Plots and compares Precision-Recall curves.
    PR-AUC is the most vital metric for highly imbalanced fraud datasets.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = ["#2563EB", "#059669", "#D97706", "#7C3AED", "#DC2626", "#0891B2", "#4F46E5"]
    
    idx = 0
    for name, y_prob in models_dict.items():
        if y_prob is None:
            continue
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        color = palette[idx % len(palette)]
        ax.plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.3f})", color=color, linewidth=2.0)
        idx += 1

    baseline = (y_true == 1).mean()
    ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.7, label=f"No-Skill Baseline ({baseline:.3f})")
    ax.set_title("Precision-Recall (PR) Curve Comparison\n(Primary Benchmark for Fraud Imbalance)", pad=15, fontweight="bold")
    ax.set_xlabel("Recall (Fraud Detected)", fontweight="bold")
    ax.set_ylabel("Precision (Flag Accuracy)", fontweight="bold")
    ax.legend(loc="lower left", frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    if static_path:
        shutil.copy(output_path, static_path)
    plt.close(fig)


def plot_feature_importance(model, feature_names: list, output_path: str, static_path: str = None):
    """Plots feature importance if model supports feature_importances_."""
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps"):
        clf = model.named_steps.get("classifier")
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_

    if importances is None:
        print("[Evaluate] Model does not provide native feature_importances_. Skipping plot.")
        return

    # Truncate or match feature names
    if len(feature_names) != len(importances):
        feature_names = [f"Feature_{i}" for i in range(len(importances))]

    df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df_imp = df_imp.sort_values(by="Importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=df_imp, y="Feature", x="Importance", palette="Blues_r", ax=ax, edgecolor="#1E293B", linewidth=0.8)
    ax.set_title("Top 15 Feature Importances", pad=15, fontweight="bold")
    ax.set_xlabel("Importance Weight", fontweight="bold")
    ax.set_ylabel("Feature Name", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    if static_path:
        shutil.copy(output_path, static_path)
    plt.close(fig)


def save_evaluation_artifacts(
    y_test, y_pred, y_prob, model_name: str,
    models_probabilities: dict,
    model_comparison_df: pd.DataFrame,
    feature_names: list = None,
    best_estimator = None,
    output_dir: str = "results",
    static_dir: str = "static/images"
):
    """Generates and writes all evaluation artifacts, metrics, and plots."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)

    metrics = calculate_metrics(y_test, y_pred, y_prob)

    # 1. Save evaluation_metrics.json
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[Evaluate] Saved metrics to {metrics_path}")

    # 2. Save classification_report.txt
    report_text = classification_report(y_test, y_pred, target_names=["Legitimate (0)", "Fraudulent (1)"])
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"=== CLASSIFICATION REPORT FOR {model_name.upper()} ===\n\n")
        f.write(report_text)
        f.write("\n\nConfusion Matrix Breakdown:\n")
        f.write(f"True Positives (Frauds Caught): {metrics['true_positives']}\n")
        f.write(f"False Positives (Legitimate Flagged): {metrics['false_positives']}\n")
        f.write(f"False Negatives (Frauds Missed): {metrics['false_negatives']}\n")
        f.write(f"True Negatives (Legitimate Cleared): {metrics['true_negatives']}\n")
    print(f"[Evaluate] Saved classification report to {report_path}")

    # 3. Save model_comparison.csv
    csv_path = os.path.join(output_dir, "model_comparison.csv")
    model_comparison_df.to_csv(csv_path, index=False)
    print(f"[Evaluate] Saved comparison table to {csv_path}")

    # 4. Save Plots
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    cm_static = os.path.join(static_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_test, y_pred, model_name, cm_path, cm_static)

    roc_path = os.path.join(output_dir, "roc_curve.png")
    roc_static = os.path.join(static_dir, "roc_curve.png")
    plot_roc_curve(models_probabilities, y_test, roc_path, roc_static)

    pr_path = os.path.join(output_dir, "precision_recall_curve.png")
    pr_static = os.path.join(static_dir, "precision_recall_curve.png")
    plot_precision_recall_curve(models_probabilities, y_test, pr_path, pr_static)

    if best_estimator is not None and feature_names is not None:
        fi_path = os.path.join(output_dir, "feature_importance.png")
        fi_static = os.path.join(static_dir, "feature_importance.png")
        plot_feature_importance(best_estimator, feature_names, fi_path, fi_static)

    print("[Evaluate] All evaluation plots successfully generated and archived.")
    return metrics
