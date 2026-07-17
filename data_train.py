import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

# =====================================================================
# 1. SETUP & DATA LOADING
# =====================================================================
FILE_PATH = "Dataset for Data Analytics - Sheet1.csv"

META_COLUMNS = ['OrderID', 'CustomerID']
NUMERIC_FEATURES = ['Quantity', 'UnitPrice', 'ItemsInCart', 'TotalPrice']
# Enforcing categorical contexts to look for deeper hidden pattern signals
CATEGORICAL_FEATURES = ['Product', 'PaymentMethod', 'CouponCode', 'ReferralSource']
TARGET_COLUMN = 'OrderStatus'

try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"Error: File not found at '{FILE_PATH}'")
    exit()

# Clean records missing crucial targets or tracking information
df = df.dropna(subset=[TARGET_COLUMN] + META_COLUMNS)

X = df[META_COLUMNS + NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET_COLUMN]

# Split 80% Train, 20% Test
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_train = X_train_raw[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
X_test = X_test_raw[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

# =====================================================================
# 2. MACHINE LEARNING PIPELINE
# =====================================================================
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, NUMERIC_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES)
    ]
)

# Using balanced class weights to compensate for uneven mock target data distributions
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42))
])

# Run Pipeline Execution
print("Training Optimized ML Pipeline Model...")
pipeline.fit(X_train, y_train)

# Predict Test Records
y_pred = pipeline.predict(X_test)

# =====================================================================
# 3. SEPARATING TABLES AND DISPLAYING ALL ROWS
# =====================================================================
results_df = X_test_raw[META_COLUMNS].copy()
results_df['Actual_Status'] = y_test
results_df['Predicted_Status'] = y_pred

unique_statuses = y.unique()

print("\n" + "="*70)
print("             GENERATING FILTERED BREAKDOWN TABLES (ALL ROWS)")
print("="*70)

for status in unique_statuses:
    status_table = results_df[results_df['Predicted_Status'] == status]
    
    print(f"\n[TABLE: Predicted Status = {status.upper()}]")
    print(f"Total Rows Displayed: {len(status_table)}")
    print("-" * 60)
    print(f"{'ORDER ID':<18} | {'CUSTOMER ID':<18} | {'ACTUAL STATUS':<18}")
    print("-" * 60)
    
    for _, row in status_table.iterrows():
        print(f"{str(row['OrderID']):<18} | {str(row['CustomerID']):<18} | {str(row['Actual_Status']):<18}")
        
    print("-" * 60)
    output_filename = f"Predicted_{status}_Orders.csv"
    status_table.to_csv(output_filename, index=False)
    print(f"💾 Saved {len(status_table)} rows to file: {output_filename}")
    print("-" * 60)

# =====================================================================
# 4. FINAL PRECISION SCOREBOARD
# =====================================================================
print("\n" + "="*70)
print("                     FINAL PRECISION SCOREBOARD")
print("="*70)

classes = sorted(unique_statuses)
per_class_precision = precision_score(y_test, y_pred, average=None, labels=classes, zero_division=0)

print("Individual Target Category Precisions:")
for idx, class_name in enumerate(classes):
    print(f" • Precision for '{class_name:<12}': {per_class_precision[idx] * 100:.2f}%")

print("-" * 70)

macro_prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
weighted_prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"Unified Global Macro Precision:     {macro_prec * 100:.2f}%")
print(f"Unified Global Weighted Precision:  {weighted_prec * 100:.2f}%")
print("="*70)
