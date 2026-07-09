"""
SMART LENDER - Loan Eligibility Prediction
============================================
Epics covered in this single script:
  Epic 1: Data Collection        -> load loan_data.csv
  Epic 2: EDA                    -> summary stats + charts (saved as PNGs)
  Epic 3: Data Preprocessing     -> missing values, encoding, scaling
  Epic 4: Model Building         -> Decision Tree, Random Forest, KNN,
                                     Gradient Boosting (XGBoost substitute)
  Epic 5: saves the BEST model + scaler + encoders for the Flask app

Run:  python3 smart_lender_pipeline.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False  # falls back to GradientBoostingClassifier (same idea, no internet needed to install)

DATA_PATH = "../data/loan_data.csv"
CHART_DIR = "charts"
MODEL_DIR = "../models"
import os
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# EPIC 1: DATA COLLECTION
# ============================================================
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print(df.head())
print(df.info())

# ============================================================
# EPIC 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\nMissing values:\n", df.isnull().sum())
print("\nTarget distribution:\n", df["Loan_Status"].value_counts())

sns.set_style("whitegrid")

# Chart 1: Loan status distribution
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="Loan_Status", palette="Set2")
plt.title("Loan Status Distribution")
plt.savefig(f"{CHART_DIR}/01_loan_status_distribution.png", bbox_inches="tight")
plt.close()

# Chart 2: Credit history vs loan status
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="Credit_History", hue="Loan_Status", palette="Set1")
plt.title("Credit History vs Loan Status")
plt.savefig(f"{CHART_DIR}/02_credit_history_vs_status.png", bbox_inches="tight")
plt.close()

# Chart 3: Applicant income distribution
plt.figure(figsize=(5, 4))
sns.histplot(df["ApplicantIncome"], kde=True, color="teal")
plt.title("Applicant Income Distribution")
plt.savefig(f"{CHART_DIR}/03_applicant_income_dist.png", bbox_inches="tight")
plt.close()

# Chart 4: Education vs Loan status
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="Education", hue="Loan_Status", palette="Set3")
plt.title("Education vs Loan Status")
plt.savefig(f"{CHART_DIR}/04_education_vs_status.png", bbox_inches="tight")
plt.close()

# Chart 5: Correlation heatmap (numeric cols)
plt.figure(figsize=(6, 5))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig(f"{CHART_DIR}/05_correlation_heatmap.png", bbox_inches="tight")
plt.close()

print(f"\n5 EDA charts saved to {CHART_DIR}")

# ============================================================
# EPIC 3: DATA PREPROCESSING
# ============================================================
data = df.copy()
data.drop(columns=["Loan_ID"], inplace=True)

# --- Handle missing values ---
cat_cols = ["Gender", "Married", "Dependents", "Self_Employed"]
for c in cat_cols:
    data[c] = data[c].fillna(data[c].mode()[0])

data["LoanAmount"] = data["LoanAmount"].fillna(data["LoanAmount"].median())
data["Loan_Amount_Term"] = data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].mode()[0])
data["Credit_History"] = data["Credit_History"].fillna(data["Credit_History"].mode()[0])

# --- Feature engineering ---
data["TotalIncome"] = data["ApplicantIncome"] + data["CoapplicantIncome"]
data["LoanAmount_log"] = np.log1p(data["LoanAmount"])
data["TotalIncome_log"] = np.log1p(data["TotalIncome"])

# --- Encode categorical variables ---
encoders = {}
label_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed",
              "Property_Area", "Loan_Status"]
for c in label_cols:
    le = LabelEncoder()
    data[c] = le.fit_transform(data[c])
    encoders[c] = le

X = data.drop(columns=["Loan_Status", "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "TotalIncome"])
y = data["Loan_Status"]

feature_columns = list(X.columns)
print("\nFinal feature columns used by the model:", feature_columns)

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# EPIC 4: MODEL BUILDING (Decision Tree, Random Forest, KNN, XGBoost)
# ============================================================
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
}
if HAS_XGB:
    models["XGBoost"] = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        eval_metric="logloss", random_state=42
    )
else:
    # XGBoost not installable offline in this sandbox -> Gradient Boosting is
    # the closest sklearn equivalent (same boosted-trees idea). Swap this
    # block for the real XGBClassifier once you install xgboost locally.
    models["XGBoost (GradientBoosting fallback)"] = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42
    )

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"\n{name}: accuracy = {acc:.4f}")
    print(classification_report(y_test, preds, target_names=["Rejected", "Approved"]))

# ============================================================
# Model comparison chart
# ============================================================
plt.figure(figsize=(6, 4))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette="viridis")
plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.xticks(rotation=20, ha="right")
plt.ylim(0, 1)
plt.savefig(f"{CHART_DIR}/06_model_comparison.png", bbox_inches="tight")
plt.close()

# ============================================================
# EPIC 5: SAVE BEST MODEL
# ============================================================
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\nBest model: {best_model_name} with accuracy {results[best_model_name]:.4f}")

with open(f"{MODEL_DIR}/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(f"{MODEL_DIR}/encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
with open(f"{MODEL_DIR}/feature_columns.json", "w") as f:
    json.dump(feature_columns, f)
with open(f"{MODEL_DIR}/results.json", "w") as f:
    json.dump({"results": results, "best_model": best_model_name}, f, indent=2)

print("\nSaved: best_model.pkl, scaler.pkl, encoders.pkl, feature_columns.json")
