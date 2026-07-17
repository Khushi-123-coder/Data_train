# Data_train
# E-Commerce Order Status Prediction Pipeline

An end-to-end Machine Learning pipeline built using **Scikit-Learn** and **Pandas** to predict order statuses based on customer cart behavior, transaction values, and acquisition channels. The pipeline handles automated data preprocessing, manages class imbalances, and outputs isolated status tables along with comprehensive performance metrics.

---

## 📌 Project Overview

This repository features a robust, automated classification pipeline engineered to analyze e-commerce datasets. It targets the `OrderStatus` column by evaluating both numeric signals (like prices and quantities) and categorical markers (such as payment channels and coupon codes). 

The application cleans input data, standardizes continuous features, encodes multi-class variables, and trains an optimized **Random Forest Classifier** with class-balancing adjustments.

---

## 🛠️ Tech Stack & Dependencies

*   **Data Manipulation:** `pandas`, `numpy`
*   **Machine Learning Framework:** `scikit-learn`
    *   *Preprocessing:* `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `ColumnTransformer`
    *   *Model:* `RandomForestClassifier` (with `class_weight='balanced'`)
    *   *Evaluation:* `precision_score`, `accuracy_score`

---

## 🗂️ Features & Pipeline Architecture

The workflow isolates tracking metadata and processes input arrays using a modular `ColumnTransformer` approach:

### 1. Data Schema & Architecture
*   **Metadata Fields (Excluded from training):** `OrderID`, `CustomerID`
*   **Numerical Transformers (Median Imputation + Standardization):**
    *   `Quantity`, `UnitPrice`, `ItemsInCart`, `TotalPrice`
*   **Categorical Transformers (Mode Imputation + One-Hot Encoding):**
    *   `Product`, `PaymentMethod`, `CouponCode`, `ReferralSource`
*   **Target Array:** `OrderStatus`

### 2. Operational Flowchart
Raw CSV Dataset
│
├──> Drop Missing Targets/Meta Records
├──> Train-Test Split (80/20 Stratified Split)
│
└──> ColumnTransformer Preprocessing
├──> Numeric: Median Imputer ──> StandardScaler
└──> Categorical: Mode Imputer ──> OneHotEncoder
│
▼
RandomForestClassifier (150 Trees, Balanced Weights)
│
▼
Predictions & Automated Output Generation
======================================================================
                     FINAL PRECISION SCOREBOARD
======================================================================
Individual Target Category Precisions:
 • Precision for 'Cancelled   ': 89.50%
 • Precision for 'Delivered   ': 92.10%
 • Precision for 'Pending     ': 85.40%
----------------------------------------------------------------------
Unified Global Macro Precision:     89.00%
Unified Global Weighted Precision:  89.33%
======================================================================
