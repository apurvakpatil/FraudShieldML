# Credit Card Fraud Detection Using Ensemble Machine Learning & Isolation Forest Anomaly Detection

An end-to-end machine learning system designed to detect fraudulent credit card transactions in real time using ensemble supervised learning algorithms, unsupervised Isolation Forest anomaly detection, robust feature engineering, an interactive Flask REST API, and a clean, modern web dashboard.

---

## 1. Project Introduction

Credit card fraud represents one of the most critical challenges facing the global banking and financial ecosystem, resulting in tens of billions of dollars in annual losses. Traditional rule-based fraud detection systems often struggle against sophisticated, rapidly evolving attack strategies and suffer from high false-positive rates that disrupt legitimate cardholders.

This project implements an intelligent, automated transaction risk-scoring system powered by **Ensemble Machine Learning**. By combining complementary algorithmic strengths (gradient boosting, tree ensembles, and stacking), the system delivers high fraud detection recall while minimizing false alarms on legitimate consumer activity.

---

## 2. Problem Statement

Financial fraud detection is characterized by **extreme class imbalance**: out of millions of daily transactions, only a minuscule fraction (typically < 1%) are fraudulent. 

When machine learning models are trained naively on highly skewed data, standard accuracy becomes a deceptive metric. A model that blindly classifies every transaction as "Legitimate" can achieve **99.42% accuracy**, yet catches **0% of fraudulent attacks**. The core challenge is to design an end-to-end pipeline that handles class imbalance, prevents data leakage, tunes the decision threshold on validation data, and prioritizes actionable metrics like **Recall**, **Precision**, and **Precision-Recall AUC (PR-AUC)**.

---

## 3. Objectives

* Analyze transaction patterns, spatial anomalies, and temporal fraud peaks through Exploratory Data Analysis (EDA).
* Build a leak-free preprocessing pipeline utilizing `scikit-learn` `ColumnTransformer` and custom feature engineering.
* Train, compare, and benchmark 8 distinct classification algorithms, focusing on ensemble methods.
* Optimize decision thresholds using validation data to maximize the balance between recall and precision.
* Save the full end-to-end pipeline (preprocessing + classifier) for zero-preprocessing inference.
* Deploy a lightweight, responsive Flask backend with RESTful API endpoints.
* Develop an intuitive, modern, professional web frontend for interactive risk evaluation and live demonstrations.

---

## 4. Dataset Description

The system is trained and evaluated strictly on the **actual uploaded credit card transaction dataset** located in the project repository:

* **File Name:** `credit_card_transactions.csv` (inside `credit_card_transactions.csv/`)
* **Total Transactions:** 1,296,675 rows
* **Total Features:** 24 raw attributes
* **Target Column:** `is_fraud` (binary classification: `0` = Legitimate, `1` = Fraudulent)
* **Class Distribution:**
  * **Legitimate Transactions (`0`):** 1,289,169 (99.421%)
  * **Fraudulent Transactions (`1`):** 7,506 (0.579%)
* **Imbalance Ratio:** ~172 legitimate transactions for every 1 fraudulent transaction.
* **Non-Predictive Columns Excluded:**
  * `Unnamed: 0` (arbitrary index)
  * `trans_num` (alphanumeric transaction hash ID)
  * `cc_num` (card number - excluded to avoid cardholder data leakage/memorization)
  * `first`, `last`, `street`, `city`, `state`, `zip` (personal identifying details)
  * `merch_zipcode` (high missingness of ~15%)
  * `unix_time` (redundant with human-readable timestamp)

---

## 5. Technologies Used

### Machine Learning & Data Processing
* **Python 3.13 / 3.11**
* **pandas** (Data manipulation and streaming chunk inspection)
* **NumPy** (Vectorized numerical computations and Haversine distance)
* **scikit-learn** (Pipelines, ColumnTransformer, Scalers, Classifiers, Ensemble models, Metrics)
* **imbalanced-learn** (Imbalanced classification techniques)
* **Matplotlib & Seaborn** (Data visualization and diagnostic curve plotting)
* **Joblib** (End-to-end model pipeline persistence)

### Backend & API
* **Flask 3.1** (Lightweight WSGI web server and REST API framework)

### Frontend
* **HTML5 & Vanilla CSS** (Responsive layout, custom design system, light theme, subtle shadows)
* **Bootstrap 5.3** (Grid system, responsive utilities, dropdowns, modal tabs)
* **FontAwesome 6** (Modern UI iconography)
* **Chart.js 4.4** (Interactive F1-score & PR-AUC comparison charts)

---

## 6. Machine Learning Workflow

```
┌─────────────────────────────────┐
│     Raw Dataset Inspection      │
│  (1,296,675 rows, is_fraud: 0.58%)
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│   Exploratory Data Analysis     │
│  (Class Imbalance, Hourly Risk, │
│   Amount Densities, Categories) │
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│       Feature Engineering       │
│  (Distance, Hour, Day, Age, Pop)│
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│   Stratified Train-Test Split   │
│   (Train: 80,000 | Test: 20,000)│
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ Preprocessing ColumnTransformer │
│ (RobustScaler + OneHotEncoder)  │
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│   Model Training & Comparison   │
│ (8 Algorithms, Validation Folds)│
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│  Validation Threshold Tuning    │
│    (Optimal threshold = 0.67)   │
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│  Held-Out Untouched Test Eval   │
│  (Confusion Matrix, ROC, PR-AUC)│
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ Pipeline & Artifact Persistence │
│ (`models/fraud_detection_model`)│
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ Flask REST API & Web Dashboard  │
└─────────────────────────────────┘
```

---

## 7. Algorithms Implemented

### Supervised Classification & Ensemble Models
1. **Logistic Regression (Baseline):** Linear classifier with `class_weight='balanced'` to establish an interpretable linear baseline.
2. **Decision Tree Classifier:** Non-linear single tree with depth constraints (`max_depth=8`) and balanced class weighting.
3. **Random Forest Classifier:** Bagging ensemble of 100 randomized decision trees operating with balanced weights to reduce variance.
4. **Extra Trees Classifier (Extremely Randomized Trees):** Randomized ensemble that randomizes split thresholds to combat noise.
5. **Gradient Boosting Classifier (Selected Supervised Model):** Sequential boosting ensemble that builds decision trees iteratively to correct prior classification residuals.
6. **HistGradientBoostingClassifier:** Modern histogram-binned gradient boosting for fast execution on large data.
7. **Soft Voting Classifier (Ensemble):** Combines the probability predictions of Random Forest, Extra Trees, and HistGradientBoosting using soft probability averaging.
8. **Stacking Classifier (Meta-Ensemble):** Uses base estimators (Random Forest, Extra Trees, HistGradientBoosting) whose out-of-fold predictions train a meta-learner (Logistic Regression).

### Unsupervised Anomaly Detection Model
9. **Isolation Forest (`scikit-learn`):** Unsupervised tree ensemble trained **exclusively on legitimate transactions** (`y_train == 0`) from the training split. It isolates anomalies by recursively partitioning feature space, where anomalous transactions require fewer splits to be isolated.
   * **Training Principle:** No fraud labels used during model fitting (zero label leakage).
   * **Score Orientation:** Uses inverted decision function ($- \text{decision\_function}(X)$) so that higher scores represent higher anomaly risk.
   * **Threshold Selection:** Operating decision threshold selected solely on held-out validation data to optimize F1-score without touching test data.

---

## 8. Explanation of Ensemble Learning

Ensemble learning combines multiple diverse machine learning models to solve a common predictive task. Rather than relying on a single algorithm, ensemble approaches leverage the principle that **diverse errors cancel out**:

* **Bagging (e.g., Random Forest):** Trains multiple trees in parallel on bootstrap subsets of data. Reduces model variance and resists overfitting.
* **Boosting (e.g., Gradient Boosting, HistGradientBoosting):** Trains trees sequentially, with each tree assigning higher focus to the difficult cases misclassified by earlier trees. Significantly improves precision and recall on minority classes.
* **Voting (Soft Voting):** Averages predicted probabilities from multiple models. Smoothes extreme outlier predictions from individual models.
* **Stacking:** Uses cross-validation predictions of base models as meta-features for a final meta-classifier.

---

## 9. Data Preprocessing & Feature Engineering

1. **Geographic Distance (`distance_km`):**
   Computed using the **Haversine formula** between cardholder coordinates (`lat`, `long`) and merchant coordinates (`merch_lat`, `merch_long`). Physical separation is a strong fraud anomaly indicator.
2. **Transaction Hour (`hour`):**
   Extracted from transaction timestamps (0-23). Analysis reveals fraud spikes between **22:00 and 04:00**.
3. **Day of Week (`day_of_week`):**
   Extracted from transaction timestamps (0-6).
4. **Cardholder Age (`age`):**
   Calculated from cardholder date of birth (`dob`) and transaction time.
5. **Transaction Amount (`amt`):**
   Scaled using `RobustScaler` (median-centered and interquartile-range scaled) to prevent extreme financial transaction outliers from skewing linear or distance calculations.
6. **Merchant Category & Gender (`category`, `gender`):**
   Processed using `OneHotEncoder(handle_unknown='ignore')` to convert categorical variables into binary vectors.

---

## 10. Model Evaluation Metrics and Their Meanings

* **Accuracy:** Overall percentage of correct classifications. *(Caution: Misleading for imbalanced fraud data).*
* **Recall (Sensitivity / True Positive Rate):** $\frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$. The fraction of actual fraud transactions successfully detected.
* **Precision:** $\frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$. The fraction of flagged transactions that were genuine frauds (minimizing cardholder annoyance).
* **F1-Score:** Harmonic mean of precision and recall: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$.
* **ROC-AUC:** Area under the Receiver Operating Characteristic curve. Measures separation ability across all thresholds.
* **PR-AUC (Average Precision):** Area under the Precision-Recall curve. **The primary gold-standard benchmark** for severe class imbalance.

---

## 11. Actual Model Comparison & Test Set Results

All candidate models were evaluated on the same cross-validation partition (Train: 60,000, Val: 20,000) and finally benchmarked on the identical untouched held-out test partition (20,000 transactions, 116 frauds).

### Full Algorithm Benchmark Comparison (`results/model_comparison.csv`):

| Algorithm | Paradigm | Recall | Precision | F1-Score | PR-AUC | ROC-AUC | Accuracy | Train Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting (Selected)** | **Supervised Boosting** | **69.57%** | **76.92%** | **0.7306** | **0.5823** | **0.9792** | **99.70%** | **19.91s** |
| Soft Voting Ensemble | Supervised Voting | 86.09% | 47.60% | 0.6130 | 0.8172 | 0.9746 | 99.38% | 14.38s |
| HistGradientBoosting | Supervised Boosting | 86.96% | 42.02% | 0.5666 | 0.8191 | 0.9901 | 99.23% | 4.15s |
| Random Forest | Supervised Bagging | 81.74% | 37.75% | 0.5165 | 0.7878 | 0.9835 | 99.12% | 2.16s |
| Extra Trees | Supervised Bagging | 80.00% | 15.23% | 0.2559 | 0.6808 | 0.9435 | 97.32% | 2.21s |
| Stacking Ensemble | Supervised Stacking | 93.04% | 14.74% | 0.2545 | 0.8048 | 0.9878 | 96.87% | 29.07s |
| Decision Tree | Supervised Tree | 88.70% | 13.11% | 0.2284 | 0.3751 | 0.9388 | 96.56% | 0.39s |
| **Isolation Forest** | **Unsupervised Anomaly** | **48.28%** | **7.03%** | **0.1227** | **0.0683** | **0.8510** | **95.99%** | **2.12s** |
| Logistic Regression | Supervised Linear | 71.30% | 3.51% | 0.0668 | 0.1421 | 0.8912 | 88.55% | 1.56s |

---

### Final Model Evaluations on 20,000 Untouched Held-Out Test Transactions

#### 1. Gradient Boosting (Supervised Ensemble — Primary Production Model)
* **Optimized Decision Threshold:** `0.67` (selected on validation partition)
* **Test Accuracy:** `99.70%`
* **Test Precision:** `79.57%`
* **Test Recall:** `63.79%`
* **Test F1-Score:** `0.7081`
* **Test ROC-AUC:** `0.9548`
* **Test PR-AUC:** `0.5767`
* **Confusion Matrix Breakdown:**
  * **True Positives (Frauds Caught):** 74
  * **False Positives (Legitimate Flagged):** 19 *(Only 0.096% false alarm rate out of 19,884 genuine transactions!)*
  * **False Negatives (Frauds Missed):** 42
  * **True Negatives (Legitimate Cleared):** 19,865

#### 2. Isolation Forest (Unsupervised Anomaly Detection — Outlier Screening Model)
* **Training Data:** Strictly legitimate transactions (`y_train == 0`, zero fraud labels used)
* **Contamination Hyperparameter:** `0.01`
* **Optimized Decision Threshold:** `-0.0215` (selected via validation partition)
* **Test Accuracy:** `95.99%`
* **Test Recall:** `48.28%` *(Successfully flagged 56 out of 116 frauds without ever seeing a fraud label!)*
* **Test Precision:** `7.03%` *(741 false alarms out of 19,884 legitimate transactions)*
* **Test F1-Score:** `0.1227`
* **Test ROC-AUC:** `0.8510` *(Robust discriminative separation for an unsupervised model)*
* **Test PR-AUC:** `0.0683`
* **Confusion Matrix Breakdown:**
  * **True Positives (Frauds Caught):** 56
  * **False Positives (Legitimate Flagged):** 741
  * **False Negatives (Frauds Missed):** 60
  * **True Negatives (Legitimate Cleared):** 19,143

#### Key Takeaways: Supervised vs. Unsupervised Trade-offs
* **Gradient Boosting** achieves superior precision (`79.57%` vs `7.03%`) and F1-score (`0.7081` vs `0.1227`) because it directly learns the boundaries separating fraudulent patterns from legitimate consumer behavior using ground-truth annotations.
* **Isolation Forest** provides high utility (`ROC-AUC: 0.8510`, `Recall: 48.28%`) in scenarios where fraud labels are delayed, incomplete, or absent (zero-day fraud detection and cold-start financial systems), identifying unusual patterns purely as topological outliers in feature space.

---

## 12. Project Folder Structure

```
AI_MINI_PROJECT_Fraude_Detection/
│
├── credit_card_transactions.csv/
│   └── credit_card_transactions.csv    # Original uploaded dataset (354 MB)
│
├── notebooks/
│   └── credit_card_fraud_eda.ipynb     # Jupyter Notebook containing full EDA steps
│
├── src/
│   ├── __init__.py                     # Package initialization
│   ├── data_loader.py                  # Dataset discovery, streaming inspection, stratified sampling
│   ├── preprocessing.py                # FeatureEngineer, RobustScaler, OneHotEncoder pipeline
│   ├── train_model.py                  # Candidate model training, validation, threshold tuning, saving
│   ├── isolation_forest.py             # Unsupervised Isolation Forest model training on legitimate data
│   ├── evaluate_model.py               # Metrics calculation, PR/ROC/CM/Importance plot generation
│   ├── predict.py                      # Multi-model inference service (GB + Isolation Forest), validation
│   └── eda.py                          # Exploratory Data Analysis visualization generator
│
├── models/
│   ├── fraud_detection_model.joblib    # Fitted end-to-end Gradient Boosting scikit-learn pipeline
│   ├── model_metadata.json             # Complete supervised training metadata and test metrics
│   ├── isolation_forest_model.joblib   # Persisted (preprocessor, isolation_forest) artifact tuple
│   └── isolation_forest_metadata.json  # Isolation Forest threshold, contamination, and evaluation metrics
│
├── results/
│   ├── model_comparison.csv            # Combined benchmark table (Supervised + Isolation Forest)
│   ├── evaluation_metrics.json         # Supervised held-out test set metrics
│   ├── classification_report.txt       # Precision, recall, and support breakdown
│   ├── confusion_matrix.png            # Confusion matrix plot
│   ├── roc_curve.png                   # Multi-model ROC comparison plot (including Isolation Forest)
│   ├── precision_recall_curve.png      # Multi-model PR comparison plot (including Isolation Forest)
│   ├── feature_importance.png          # Top 15 feature importances plot
│   ├── eda_class_distribution.png      # Class imbalance chart
│   ├── eda_amount_distribution.png     # Amount density (log scale) chart
│   ├── eda_fraud_by_hour.png           # Hourly risk curve
│   ├── eda_fraud_by_category.png       # Category risk bar chart
│   └── eda_correlation_matrix.png      # Feature correlation heatmap
│
├── templates/
│   └── index.html                      # Modern, responsive web frontend dashboard with model selection
│
├── static/
│   ├── css/
│   │   └── style.css                   # Custom light theme CSS styling
│   ├── js/
│   │   └── app.js                      # Dynamic frontend logic, API calls, Chart.js, multi-model UI
│   └── images/                         # Diagnostic and EDA charts for web presentation
│
├── tests/
│   └── test_api.py                     # Automated unit and integration test suite (15 tests)
│
├── app.py                              # Flask application backend with REST API endpoints
├── run.py                              # Application launcher entrypoint
├── requirements.txt                    # Project Python dependencies
├── .gitignore                          # Standard git ignore rules
└── README.md                           # Comprehensive documentation
```

---

## 13. Installation Instructions

1. **Clone or navigate to the project directory:**
   ```bash
   cd e:\AI_MINI_PROJECT_Fraude_Detection
   ```

2. **(Optional) Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 14. Training Instructions

To retrain the candidate models, re-evaluate the ensemble models, optimize the decision threshold, and persist new artifacts:

```bash
# Run model training pipeline with 100,000 stratified samples
python -m src.train_model --sample-size 100000 --seed 42
```

To run the Exploratory Data Analysis visualization generator:
```bash
python src/eda.py
```

---

## 15. Running the Automated Test Suite

Execute the 15 automated unit and integration tests:

```bash
python -m pytest tests/test_api.py -v
```

All 15 tests verify:
* Dataset discovery and genuine transaction extraction
* Model persistence and loading (both Gradient Boosting and Isolation Forest)
* System health reporting for both models (`/api/health`)
* Default prediction endpoint (backward-compatible Gradient Boosting inference)
* High-risk vs. low-risk fraud scenarios
* Input payload validation and bounds checking
* Explicit model selection (`"model": "isolation_forest"`)
* Anomaly score calculation and score orientation (higher score = more anomalous)
* Rejection of unsupported model names with HTTP 400
* Model comparison endpoint (`/api/comparison`) verifying no duplicates
* Model metadata integrity and web dashboard serving

---

## 16. Application Startup Instructions

Launch the local web application:

```bash
python run.py
```

Open your browser and navigate to:
**`http://127.0.0.1:5000`**

---

## 17. Instructions for Making Predictions

### Method A: Through the Web Interface
1. Open `http://127.0.0.1:5000` in your web browser.
2. Scroll to the **Transaction Risk Screening Form**.
3. Choose the **Detection Algorithm Architecture**:
   * **Gradient Boosting (Default):** Supervised ensemble outputting calibrated fraud probabilities.
   * **Isolation Forest:** Unsupervised anomaly detector outputting pure outlier anomaly scores.
4. Click the **Load Real Sample** button to choose from genuine transactions extracted from the dataset:
   * **Sample 1 / Sample 2:** Standard legitimate transactions ($10 - $75, daytime).
   * **Sample 3 / Sample 4:** High-risk fraudulent transactions ($300 - $980, late night hours).
5. Click **Analyze Transaction**.
6. Observe the live risk assessment card:
   * Status badge (**Fraud / Anomaly Alert** vs **Legitimate Transaction**).
   * Dynamic Risk Meter: Displays **Fraud Probability** (0.0 to 1.0) for Gradient Boosting or **Anomaly Score** with respect to the decision threshold (`-0.0215`) for Isolation Forest.
   * Model rationale highlighting specific contributing risk factors and feature vectors.
   * Real-time session history table tracking all past predictions with model identification.

### Method B: Via REST API

#### 1. Default Supervised Prediction (Gradient Boosting):
Send an HTTP POST request to `/api/predict`:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gradient_boosting",
    "amt": 850.00,
    "category": "shopping_net",
    "hour": 23,
    "day_of_week": 4,
    "distance_km": 280.0,
    "age": 42,
    "gender": "F",
    "city_pop": 75000
  }'
```

**Example JSON Response:**
```json
{
  "status": "success",
  "prediction": 1,
  "label": "Fraudulent Transaction",
  "score_type": "fraud_probability",
  "fraud_probability": 0.8845,
  "legit_probability": 0.1155,
  "anomaly_score": null,
  "threshold": 0.67,
  "risk_level": "Critical Risk",
  "model_name": "Gradient Boosting",
  "explanation": "The supervised ensemble model classified this transaction as POTENTIALLY FRAUDULENT. Key contributing factors: High transaction amount ($850.00), Off-peak night hour (23:00), Elevated risk merchant category 'shopping_net', Unusual distance from cardholder location (280.0 km).",
  "disclaimer": "This prediction is generated by an ensemble machine learning model for risk screening and decision support. It is not an absolute financial guarantee."
}
```

#### 2. Unsupervised Anomaly Detection (Isolation Forest):
Send an HTTP POST request with `"model": "isolation_forest"`:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model": "isolation_forest",
    "amt": 980.50,
    "category": "shopping_net",
    "hour": 2,
    "day_of_week": 5,
    "distance_km": 350.0,
    "age": 45,
    "gender": "F",
    "city_pop": 120000
  }'
```

**Example JSON Response:**
```json
{
  "status": "success",
  "prediction": 1,
  "label": "Fraudulent Transaction",
  "score_type": "anomaly_score",
  "anomaly_score": 0.0152,
  "fraud_probability": null,
  "legit_probability": null,
  "threshold": -0.0215,
  "risk_level": "Critical Risk",
  "model_name": "Isolation Forest",
  "explanation": "The unsupervised Isolation Forest model flagged this transaction as ANOMALOUS (Score: 0.0152 >= Threshold: -0.0215). The transaction exhibits feature isolation characteristics atypical of legitimate behavior. Notable risk flags: High transaction amount ($980.50), Off-peak night hour (2:00), Elevated risk merchant category 'shopping_net', Unusual distance from cardholder location (350.0 km).",
  "disclaimer": "This prediction is generated by an unsupervised Isolation Forest model for outlier risk screening. It is not an absolute financial guarantee."
}
```

#### 3. Model Benchmark Comparison Endpoint:
Retrieve benchmark metrics comparing all supervised models against Isolation Forest:

```bash
curl -X GET http://127.0.0.1:5000/api/comparison
```

---

## 18. Limitations and Future Improvements

* **Temporal Sequence Modeling:** Incorporating customer spending velocity features (e.g., number of transactions within the last 15 minutes, rolling average spend).
* **Graph Neural Networks (GNNs):** Modeling relationships between cardholders, merchants, and devices as a heterogeneous graph to catch organized fraud rings.
* **Online Adaptive Learning:** Incrementally updating decision trees to adapt to new fraud patterns without full retraining.
* **Explainable AI (SHAP/LIME):** Integrating SHAP waterfall plots for granular local feature attribution in bank compliance workflows.
# FaudeShieldML
