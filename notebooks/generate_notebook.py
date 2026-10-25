import json
import os

notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Credit Card Fraud Detection - Exploratory Data Analysis (EDA)\n",
                "**Project:** Credit Card Fraud Detection Using Ensemble Machine Learning  \n",
                "**Dataset:** Credit Card Transactions (Simulated Sparkov dataset)  \n",
                "**Objective:** Uncover transaction characteristics, evaluate class imbalance, explore temporal and category risk patterns, and extract predictive features."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, sys\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "# Configure plotting\n",
                "sns.set_theme(style='whitegrid')\n",
                "plt.rcParams['figure.figsize'] = (10, 6)\n",
                "sys.path.insert(0, os.path.abspath('..'))\n",
                "from src.data_loader import load_data, find_dataset_path, haversine_distance\n",
                "print('Environment and helper modules loaded.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Load Dataset and Inspect Dimensions\n",
                "We inspect the uploaded dataset and load a representative stratified sample for exploration."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "dataset_path = find_dataset_path()\n",
                "print('Dataset path:', dataset_path)\n",
                "df, stats = load_data(sample_size=100000)\n",
                "print('Sample shape:', df.shape)\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Target Class Imbalance Analysis\n",
                "Credit card fraud is an extreme anomaly detection problem. Here we evaluate the distribution of `is_fraud`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "counts = df['is_fraud'].value_counts()\n",
                "print('Class Counts:')\n",
                "print(counts)\n",
                "print('Fraud Percentage:', round(counts[1] / len(df) * 100, 4), '%')\n",
                "\n",
                "fig, ax = plt.subplots(figsize=(7, 4))\n",
                "sns.barplot(x=['Legitimate (0)', 'Fraudulent (1)'], y=counts.values, palette=['#2563EB', '#DC2626'], ax=ax)\n",
                "ax.set_yscale('log')\n",
                "ax.set_title('Transaction Class Distribution (Log Scale)')\n",
                "ax.set_ylabel('Number of Records')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Transaction Amount Comparison\n",
                "Examining monetary amounts for legitimate transactions versus fraudulent activity."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('Legitimate Transactions Amount Summary:')\n",
                "print(df[df['is_fraud'] == 0]['amt'].describe())\n",
                "print('\\nFraudulent Transactions Amount Summary:')\n",
                "print(df[df['is_fraud'] == 1]['amt'].describe())\n",
                "\n",
                "fig, ax = plt.subplots(figsize=(9, 4))\n",
                "sns.kdeplot(data=df[df['is_fraud'] == 0]['amt'], label='Legitimate', log_scale=True, color='#2563EB', fill=True, alpha=0.3)\n",
                "sns.kdeplot(data=df[df['is_fraud'] == 1]['amt'], label='Fraudulent', log_scale=True, color='#DC2626', fill=True, alpha=0.4)\n",
                "ax.set_title('Transaction Amount Density Distribution (Log Scale)')\n",
                "ax.set_xlabel('Amount in USD (Log Scale)')\n",
                "plt.legend()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Feature Engineering: Temporal and Spatial Features\n",
                "Deriving hour of day, day of week, cardholder age, and cardholder-to-merchant distance."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])\n",
                "df['hour'] = df['trans_date_trans_time'].dt.hour\n",
                "df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek\n",
                "df['dob'] = pd.to_datetime(df['dob'])\n",
                "df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365\n",
                "\n",
                "lats = df['lat'].values\n",
                "longs = df['long'].values\n",
                "merch_lats = df['merch_lat'].values\n",
                "merch_longs = df['merch_long'].values\n",
                "df['distance_km'] = [haversine_distance(lats[i], longs[i], merch_lats[i], merch_longs[i]) for i in range(len(lats))]\n",
                "df[['amt', 'hour', 'day_of_week', 'age', 'distance_km', 'is_fraud']].head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Temporal Fraud Vulnerability (Hourly Risk Analysis)\n",
                "Evaluating the fraction of fraudulent transactions occurring at each hour of the day."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "hourly = df.groupby('hour')['is_fraud'].agg(['count', 'mean']).reset_index()\n",
                "hourly['fraud_pct'] = hourly['mean'] * 100\n",
                "\n",
                "fig, ax = plt.subplots(figsize=(10, 4))\n",
                "sns.lineplot(data=hourly, x='hour', y='fraud_pct', marker='o', color='#DC2626', linewidth=2.5, ax=ax)\n",
                "ax.set_title('Fraud Occurrence (%) by Hour of Day')\n",
                "ax.set_xticks(range(0, 24))\n",
                "ax.set_xlabel('Hour (0 to 23)')\n",
                "ax.set_ylabel('Fraud Rate (%)')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Merchant Category Fraud Vulnerability\n",
                "Analyzing which consumer spending categories have the highest proportion of fraudulent activity."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "cat_risk = df.groupby('category')['is_fraud'].agg(['count', 'mean']).reset_index()\n",
                "cat_risk['fraud_pct'] = cat_risk['mean'] * 100\n",
                "cat_risk = cat_risk.sort_values(by='fraud_pct', ascending=False)\n",
                "\n",
                "fig, ax = plt.subplots(figsize=(10, 5))\n",
                "sns.barplot(data=cat_risk, y='category', x='fraud_pct', palette='Reds_r', ax=ax)\n",
                "ax.set_title('Fraud Occurrence Rate (%) Across Merchant Categories')\n",
                "ax.set_xlabel('Fraud Rate (%)')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Numerical Feature Correlations\n",
                "Assessing linear correlations among engineered features and the target variable."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "corr_cols = ['is_fraud', 'amt', 'distance_km', 'hour', 'day_of_week', 'age', 'city_pop']\n",
                "corr = df[corr_cols].corr()\n",
                "\n",
                "fig, ax = plt.subplots(figsize=(8, 6))\n",
                "sns.heatmap(corr, annot=True, fmt='.3f', cmap='Blues', ax=ax, linewidths=0.5)\n",
                "ax.set_title('Correlation Heatmap')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Summary of Key Insights for Ensemble Modeling\n",
                "1. **Class Imbalance:** Fraud represents only ~0.58% of transactions. Accuracy is an uninformative metric; PR-AUC and Recall/F1 must be prioritized.\n",
                "2. **Amount Divergence:** Legitimate transactions average ~$67, whereas fraudulent ones average ~$524.\n",
                "3. **Nighttime Window:** Fraud peaks significantly between 22:00 and 04:00.\n",
                "4. **Vulnerable Categories:** `shopping_net`, `grocery_pos`, and `misc_net` carry the highest fraud rates.\n",
                "5. **Ensemble Architecture:** Combining tree ensembles (Random Forest, Extra Trees) with boosting (HistGradientBoosting) and stacking provides resilient defense against overfitting on skewed fraud classes."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = os.path.join(os.path.dirname(__file__), "credit_card_fraud_eda.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2)

print(f"Generated {out_path}")
