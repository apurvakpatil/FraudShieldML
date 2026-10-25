"""
Model Training and Ensemble Learning Pipeline for Credit Card Fraud Detection
Trains individual classifiers and ensemble models (Random Forest, Extra Trees,
Gradient Boosting, HistGradientBoosting, Soft Voting, Stacking), optimizes thresholds,
evaluates on held-out test data, and persists the best end-to-end pipeline.
"""

import os
import sys
import json
import time
import argparse
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import load_data, find_dataset_path
from src.preprocessing import (
    build_full_preprocessing_pipeline,
    prepare_train_test_data,
    ALL_MODEL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
)
from src.evaluate_model import (
    calculate_metrics,
    find_optimal_threshold,
    save_evaluation_artifacts,
)


def get_candidate_models(random_state: int = 42) -> dict:
    """
    Initializes diverse classification algorithms and ensemble architectures
    configured specifically for imbalanced fraud detection.
    """
    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs"
    )
    dt = DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=8,
        min_samples_leaf=20,
        random_state=random_state
    )
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        max_depth=12,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1
    )
    et = ExtraTreesClassifier(
        n_estimators=100,
        class_weight="balanced",
        max_depth=12,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1
    )
    hgb = HistGradientBoostingClassifier(
        class_weight="balanced",
        max_iter=100,
        max_depth=8,
        random_state=random_state
    )
    gb = GradientBoostingClassifier(
        n_estimators=80,
        max_depth=5,
        subsample=0.8,
        random_state=random_state
    )

    # Ensemble 1: Soft Voting combining complementary tree + gradient + linear approaches
    voting = VotingClassifier(
        estimators=[
            ("rf", rf),
            ("et", et),
            ("hgb", hgb),
        ],
        voting="soft",
        n_jobs=-1
    )

    # Ensemble 2: Stacking Classifier using complementary base models and logistic meta-learner
    stacking = StackingClassifier(
        estimators=[
            ("rf", rf),
            ("et", et),
            ("hgb", hgb),
        ],
        final_estimator=LogisticRegression(class_weight="balanced", random_state=random_state),
        cv=3,
        n_jobs=-1
    )

    return {
        "Logistic Regression": lr,
        "Decision Tree": dt,
        "Random Forest": rf,
        "Extra Trees": et,
        "Gradient Boosting": gb,
        "HistGradientBoosting": hgb,
        "Soft Voting Ensemble": voting,
        "Stacking Ensemble": stacking,
    }


def train_and_evaluate_all(
    sample_size: int = 100000,
    random_state: int = 42,
    output_dir: str = "results",
    model_dir: str = "models",
    static_dir: str = "static/images"
):
    """
    Orchestrates the entire training, validation, threshold optimization,
    model selection, untouched test set evaluation, and model artifact persistence.
    """
    start_total_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)

    print("\n" + "="*70)
    print("  CREDIT CARD FRAUD DETECTION: ENSEMBLE MODEL TRAINING PIPELINE")
    print("="*70)

    # 1. Load Data
    dataset_path = find_dataset_path()
    df, stats = load_data(sample_size=sample_size, random_state=random_state, dataset_path=dataset_path)

    # 2. Partition into untouched Test Set (20%) and Training Data (80%)
    X_train, X_test, y_train, y_test = prepare_train_test_data(df, test_size=0.2, random_state=random_state)

    # 3. Create a Validation split from Training Data for fair model comparison and threshold tuning
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.25,  # 0.25 of 80% = 20% validation, leaving 60% pure train
        random_state=random_state,
        stratify=y_train
    )
    print(f"[Training] Cross-validation partition - Train: {len(X_tr):,}, Val: {len(X_val):,}")

    # 4. Preprocess features strictly fitted on X_tr
    print("[Training] Fitting preprocessing pipeline on pure training partition...")
    preprocessor = build_full_preprocessing_pipeline()
    X_tr_proc = preprocessor.fit_transform(X_tr)
    X_val_proc = preprocessor.transform(X_val)

    # Retrieve processed feature names for feature importance
    cat_encoder = preprocessor.named_steps["transform"].named_transformers_["cat"].named_steps["encoder"]
    encoded_cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    processed_feature_names = NUMERICAL_FEATURES + encoded_cat_names

    # 5. Train & Compare all Candidate Models on Validation Partition
    candidates = get_candidate_models(random_state=random_state)
    comparison_records = []
    val_probabilities = {}
    fitted_val_models = {}

    print("\n" + "-"*70)
    print(f"{'Model Name':<25} | {'Val Recall':<10} | {'Val Prec':<9} | {'Val F1':<8} | {'Val PR-AUC':<10} | {'Time (s)':<8}")
    print("-"*70)

    for name, clf in candidates.items():
        t0 = time.time()
        clf.fit(X_tr_proc, y_tr)
        t_elapsed = round(time.time() - t0, 2)
        fitted_val_models[name] = clf

        # Probabilities
        if hasattr(clf, "predict_proba"):
            y_prob_val = clf.predict_proba(X_val_proc)[:, 1]
        elif hasattr(clf, "decision_function"):
            dfunc = clf.decision_function(X_val_proc)
            y_prob_val = (dfunc - dfunc.min()) / (dfunc.max() - dfunc.min() + 1e-8)
        else:
            y_prob_val = clf.predict(X_val_proc).astype(float)

        val_probabilities[name] = y_prob_val
        y_pred_val = (y_prob_val >= 0.5).astype(int)

        m = calculate_metrics(y_val, y_pred_val, y_prob_val)
        comparison_records.append({
            "Model": name,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1-Score": m["f1_score"],
            "ROC-AUC": m["roc_auc"],
            "PR-AUC": m["pr_auc"],
            "Training Time (s)": t_elapsed
        })

        print(f"{name:<25} | {m['recall']:<10.4f} | {m['precision']:<9.4f} | {m['f1_score']:<8.4f} | {m['pr_auc']:<10.4f} | {t_elapsed:<8.2f}")

    print("-"*70)
    comparison_df = pd.DataFrame(comparison_records).sort_values(by="F1-Score", ascending=False).reset_index(drop=True)

    # 6. Select Best Performing Model based on Validation F1-Score & PR-AUC
    # Prioritize top ensemble (Voting or Stacking or Random Forest)
    best_model_name = comparison_df.iloc[0]["Model"]
    print(f"\n[Model Selection] Best performing model on validation set: '{best_model_name}'")
    print(f"[Model Selection] Validation F1-Score: {comparison_df.iloc[0]['F1-Score']:.4f}, PR-AUC: {comparison_df.iloc[0]['PR-AUC']:.4f}")

    # 7. Tune Decision Threshold on Validation Data for Best Model
    best_val_clf = fitted_val_models[best_model_name]
    best_val_prob = val_probabilities[best_model_name]
    optimal_threshold = find_optimal_threshold(y_val, best_val_prob, metric="f1")
    print(f"[Threshold Optimization] Optimal decision threshold on validation data: {optimal_threshold} (Default was 0.50)")

    # 8. Retrain Best Model on Complete Training Set (X_train, y_train) inside End-to-End Pipeline
    print(f"\n[Final Training] Retraining '{best_model_name}' on all {len(X_train):,} training records with full pipeline...")
    final_preprocessor = build_full_preprocessing_pipeline()
    
    # Re-instantiate a clean instance of best model
    fresh_candidates = get_candidate_models(random_state=random_state)
    final_classifier = fresh_candidates[best_model_name]

    final_pipeline = Pipeline([
        ("preprocessor", final_preprocessor),
        ("classifier", final_classifier)
    ])

    final_pipeline.fit(X_train, y_train)
    print("[Final Training] Pipeline fitted successfully.")

    # 9. Evaluate on Held-Out UNTOUCHED Test Set
    print("\n" + "="*70)
    print("  FINAL EVALUATION ON HELD-OUT UNTOUCHED TEST SET")
    print("="*70)

    # Predictions using optimal threshold
    if hasattr(final_pipeline, "predict_proba"):
        y_prob_test = final_pipeline.predict_proba(X_test)[:, 1]
    elif hasattr(final_pipeline, "decision_function"):
        dfunc = final_pipeline.decision_function(X_test)
        y_prob_test = (dfunc - dfunc.min()) / (dfunc.max() - dfunc.min() + 1e-8)
    else:
        y_prob_test = final_pipeline.predict(X_test).astype(float)

    y_pred_test = (y_prob_test >= optimal_threshold).astype(int)

    # Compute comparison test curves for top models
    test_models_probabilities = {best_model_name: y_prob_test}
    # Also evaluate other top contenders for comparison curves
    for other_name in ["Random Forest", "HistGradientBoosting", "Soft Voting Ensemble", "Stacking Ensemble"]:
        if other_name in fitted_val_models and other_name != best_model_name:
            # Transform test set using the fitted validation preprocessor for curve plotting
            proc_test = preprocessor.transform(X_test)
            other_clf = fitted_val_models[other_name]
            if hasattr(other_clf, "predict_proba"):
                test_models_probabilities[other_name] = other_clf.predict_proba(proc_test)[:, 1]

    # 10. Save Evaluation Artifacts and Plots
    test_metrics = save_evaluation_artifacts(
        y_test=y_test,
        y_pred=y_pred_test,
        y_prob=y_prob_test,
        model_name=best_model_name,
        models_probabilities=test_models_probabilities,
        model_comparison_df=comparison_df,
        feature_names=processed_feature_names,
        best_estimator=final_classifier,
        output_dir=output_dir,
        static_dir=static_dir
    )

    print("\nTest Set Performance Summary:")
    print(f"  Accuracy:         {test_metrics['accuracy']:.4f}")
    print(f"  Precision:        {test_metrics['precision']:.4f}")
    print(f"  Recall:           {test_metrics['recall']:.4f}")
    print(f"  F1-Score:         {test_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:          {test_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:           {test_metrics['pr_auc']:.4f}")
    print(f"  True Positives:   {test_metrics['true_positives']} (Frauds Correctly Flagged)")
    print(f"  False Positives:  {test_metrics['false_positives']} (Legit Incorrectly Flagged)")
    print(f"  False Negatives:  {test_metrics['false_negatives']} (Frauds Missed)")
    print(f"  True Negatives:   {test_metrics['true_negatives']} (Legit Cleared)")

    # 11. Save the End-to-End Model Artifact
    model_save_path = os.path.join(model_dir, "fraud_detection_model.joblib")
    joblib.dump(final_pipeline, model_save_path)
    print(f"\n[Persistence] Final trained pipeline saved to: {model_save_path}")

    # 12. Save Comprehensive Model Metadata
    metadata = {
        "model_name": best_model_name,
        "model_version": "1.0.0",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_file": os.path.basename(dataset_path),
        "target_column": TARGET_COLUMN,
        "input_features": ALL_MODEL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "total_analyzed_samples": len(df),
        "random_seed": random_state,
        "selected_threshold": optimal_threshold,
        "validation_metrics": comparison_df.to_dict(orient="records"),
        "test_metrics": test_metrics,
        "pipeline_steps": [name for name, _ in final_pipeline.steps],
    }

    metadata_path = os.path.join(model_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[Persistence] Model metadata saved to: {metadata_path}")

    total_time = round(time.time() - start_total_time, 2)
    print(f"\n[Pipeline Finished] Complete execution completed in {total_time}s.")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Credit Card Fraud Detection Models")
    parser.add_argument("--sample-size", type=int, default=100000, help="Stratified sample size (default: 100000)")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed")
    args = parser.parse_args()

    train_and_evaluate_all(sample_size=args.sample_size, random_state=args.seed)
