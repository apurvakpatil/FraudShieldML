# Credit Card Fraud Detection Using Ensemble Machine Learning

An end-to-end machine learning system designed to detect fraudulent credit card transactions in real time using ensemble learning algorithms, robust feature engineering, an interactive Flask REST API, and a clean, modern web dashboard.

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

1. **Logistic Regression (Baseline):** Linear classifier with `class_weight='balanced'` to establish an interpretable linear baseline.
2. **Decision Tree Classifier:** Non-linear single tree with depth constraints (`max_depth=8`) and balanced class weighting.
3. **Random Forest Classifier:** Bagging ensemble of 100 randomized decision trees operating with balanced weights to reduce variance.
4. **Extra Trees Classifier (Extremely Randomized Trees):** Randomized ensemble that randomizes split thresholds to combat noise.
5. **Gradient Boosting Classifier:** Sequential boosting ensemble that builds decision trees iteratively to correct prior classification residuals.
6. **HistGradientBoostingClassifier:** Modern histogram-binned gradient boosting for fast execution on large data.
7. **Soft Voting Classifier (Ensemble):** Combines the probability predictions of Random Forest, Extra Trees, and HistGradientBoosting using soft probability averaging.
8. **Stacking Classifier (Meta-Ensemble):** Uses base estimators (Random Forest, Extra Trees, HistGradientBoosting) whose out-of-fold predictions train a meta-learner (Logistic Regression).

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

### Candidate Models Comparison (Validation Set):

| Algorithm | Val Recall | Val Precision | Val F1-Score | Val PR-AUC | Val ROC-AUC | Val Accuracy | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting (Selected)** | **69.57%** | **76.19%** | **0.7273** | **0.5817** | **0.9792** | **99.70%** | **10.69s** |
| Random Forest | 77.39% | 56.33% | 0.6520 | 0.7730 | 0.9862 | 99.52% | 0.75s |
| Soft Voting Ensemble | 84.35% | 39.27% | 0.5359 | 0.7943 | 0.9713 | 99.16% | 6.14s |
| HistGradientBoosting | 87.83% | 27.45% | 0.4182 | 0.7602 | 0.9848 | 98.59% | 2.57s |
| Extra Trees | 80.00% | 15.23% | 0.2559 | 0.6808 | 0.9435 | 97.32% | 0.53s |
| Decision Tree | 88.70% | 13.11% | 0.2284 | 0.3751 | 0.9388 | 96.56% | 0.21s |
| Stacking Ensemble | 93.91% | 11.79% | 0.2095 | 0.7887 | 0.9881 | 95.93% | 17.09s |
| Logistic Regression | 71.30% | 3.49% | 0.0666 | 0.1419 | 0.8914 | 88.50% | 0.43s |

### Final Model Evaluation on 20,000 Untouched Held-Out Test Transactions:

* **Selected Model:** Gradient Boosting Pipeline
* **Optimized Decision Threshold:** `0.67` (selected via validation set)
* **Test Accuracy:** `99.68%`
* **Test Precision:** `77.66%`
* **Test Recall:** `62.93%`
* **Test F1-Score:** `0.6952`
* **Test ROC-AUC:** `0.9445`
* **Test PR-AUC:** `0.5249`
* **Confusion Matrix Breakdown:**
  * **True Positives (Frauds Caught):** 73
  * **False Positives (Legitimate Flagged):** 21 *(Only 0.10% false alarm rate!)*
  * **False Negatives (Frauds Missed):** 43
  * **True Negatives (Legitimate Cleared):** 19,863

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
│   ├── evaluate_model.py               # Metrics calculation, PR/ROC/CM/Importance plot generation
│   ├── predict.py                      # Production inference service, validation, explanations
│   └── eda.py                          # Exploratory Data Analysis visualization generator
│
├── models/
│   ├── fraud_detection_model.joblib    # Fitted end-to-end scikit-learn pipeline artifact
│   └── model_metadata.json             # Complete model training metadata, features, and metrics
│
├── results/
│   ├── model_comparison.csv            # Algorithm performance comparison table
│   ├── evaluation_metrics.json         # Held-out test set metrics
│   ├── classification_report.txt       # Precision, recall, and support breakdown
│   ├── confusion_matrix.png            # Confusion matrix plot
│   ├── roc_curve.png                   # ROC curves comparison plot
│   ├── precision_recall_curve.png      # PR curves comparison plot
│   ├── feature_importance.png          # Top 15 feature importances plot
│   ├── eda_class_distribution.png      # Class imbalance chart
│   ├── eda_amount_distribution.png     # Amount density (log scale) chart
│   ├── eda_fraud_by_hour.png           # Hourly risk curve
│   ├── eda_fraud_by_category.png       # Category risk bar chart
│   └── eda_correlation_matrix.png      # Feature correlation heatmap
│
├── templates/
│   └── index.html                      # Modern, responsive web frontend dashboard
│
├── static/
│   ├── css/
│   │   └── style.css                   # Custom light theme CSS styling
│   ├── js/
│   │   └── app.js                      # Dynamic frontend logic, API calls, Chart.js
│   └── images/                         # Diagnostic and EDA charts for web presentation
│
├── tests/
│   └── test_api.py                     # Automated unit and integration test suite (10 tests)
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

Execute the 10 automated unit and integration tests:

```bash
python -m pytest tests/test_api.py -v
```

All tests verify dataset discovery, model persistence, sample transactions, inference logic, and Flask endpoints.

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
3. Click the **Load Real Sample** button to choose from genuine transactions extracted from the dataset:
   * **Sample 1 / Sample 2:** Standard legitimate transactions ($10 - $75, daytime).
   * **Sample 3 / Sample 4:** High-risk fraudulent transactions ($300 - $980, late night hours).
4. Click **Analyze Transaction**.
5. Observe the live risk assessment card:
   * Status badge (**Fraud Alert** vs **Legitimate Transaction**).
   * Fraud Probability meter with risk level coloring.
   * Model rationale highlighting specific contributing risk factors.
   * Real-time session history table tracking all past predictions.

### Method B: Via REST API
Send an HTTP POST request to `/api/predict`:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "amt": 850.00,
      "category": "shopping_net",
      "hour": 23,
      "day_of_week": 4,
      "distance_km": 280.0,
      "age": 42,
      "gender": "F",
      "city_pop": 75000
    }
  }'
```

**Example JSON Response:**
```json
{
  "status": "success",
  "prediction": 1,
  "label": "Fraudulent Transaction",
  "fraud_probability": 0.8845,
  "legit_probability": 0.1155,
  "threshold": 0.67,
  "risk_level": "High Risk",
  "model_name": "Gradient Boosting",
  "explanation": "The ensemble model classified this transaction as POTENTIALLY FRAUDULENT. Key contributing factors: High transaction amount ($850.00), Off-peak night hour (23:00), Elevated risk merchant category 'shopping_net', Unusual distance from cardholder location (280.0 km).",
  "disclaimer": "This prediction is generated by an ensemble machine learning model for risk screening and decision support. It is not an absolute financial guarantee."
}
```

---

## 18. Limitations and Future Improvements

* **Temporal Sequence Modeling:** Incorporating customer spending velocity features (e.g., number of transactions within the last 15 minutes, rolling average spend).
* **Graph Neural Networks (GNNs):** Modeling relationships between cardholders, merchants, and devices as a heterogeneous graph to catch organized fraud rings.
* **Online Adaptive Learning:** Incrementally updating decision trees to adapt to new fraud patterns without full retraining.
* **Explainable AI (SHAP/LIME):** Integrating SHAP waterfall plots for granular local feature attribution in bank compliance workflows.
#   F a u d e S h i e l d M L  
 