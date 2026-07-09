"""
Smart Lender - Dataset Generator
=================================
Generates a synthetic loan-applicant dataset that matches the schema of the
classic "Loan Prediction" dataset used in most AI/ML internship projects
(Loan_ID, Gender, Married, Dependents, Education, Self_Employed,
ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
Credit_History, Property_Area, Loan_Status).

NOTE: This environment has no internet access, so the real Kaggle dataset
could not be downloaded. This script creates a synthetic dataset with the
SAME columns and realistic value distributions, so all the downstream code
(EDA / preprocessing / model training) works exactly the same way.

If you have internet access, just download the real dataset from Kaggle
("Loan Prediction Problem Dataset" by ninzaami) and drop the CSV in this
folder as loan_data.csv with the same column names - no other code changes
needed.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 614  # same size as the original Kaggle loan prediction train set

gender = np.random.choice(["Male", "Female"], size=N, p=[0.81, 0.19])
married = np.random.choice(["Yes", "No"], size=N, p=[0.65, 0.35])
dependents = np.random.choice(["0", "1", "2", "3+"], size=N, p=[0.58, 0.17, 0.17, 0.08])
education = np.random.choice(["Graduate", "Not Graduate"], size=N, p=[0.78, 0.22])
self_employed = np.random.choice(["Yes", "No"], size=N, p=[0.14, 0.86])
property_area = np.random.choice(["Urban", "Semiurban", "Rural"], size=N, p=[0.33, 0.38, 0.29])

applicant_income = np.random.gamma(shape=2.0, scale=2500, size=N).astype(int) + 1500
coapplicant_income = np.where(
    married == "Yes",
    np.random.gamma(shape=1.5, scale=1200, size=N).astype(int),
    0
)
loan_amount = (0.02 * applicant_income + 0.015 * coapplicant_income
               + np.random.normal(0, 40, size=N)).clip(9, 700).astype(int)
loan_amount_term = np.random.choice(
    [360, 180, 120, 240, 84, 300, 60, 36],
    size=N, p=[0.72, 0.09, 0.05, 0.05, 0.03, 0.03, 0.02, 0.01]
)
credit_history = np.random.choice([1.0, 0.0], size=N, p=[0.84, 0.16])

# introduce some realistic missing values (like the real dataset has)
def add_missing(arr, frac):
    arr = arr.astype(object)
    idx = np.random.choice(len(arr), size=int(len(arr) * frac), replace=False)
    arr[idx] = np.nan
    return arr

gender = add_missing(gender, 0.02)
married = add_missing(married, 0.005)
dependents = add_missing(dependents, 0.025)
self_employed = add_missing(self_employed, 0.05)
loan_amount = add_missing(loan_amount.astype(float), 0.035)
loan_amount_term = add_missing(loan_amount_term.astype(float), 0.023)
credit_history = add_missing(credit_history, 0.08)

# Target variable driven by a realistic underlying rule + noise
credit_history_filled = pd.Series(credit_history).fillna(1.0).astype(float).values
loan_amount_filled = pd.Series(loan_amount).fillna(np.nanmedian(pd.Series(loan_amount).astype(float))).astype(float).values

score = (
    2.4 * credit_history_filled
    + 0.00035 * applicant_income
    + 0.00025 * coapplicant_income
    - 0.006 * loan_amount_filled
    + np.where(education == "Graduate", 0.35, -0.1)
    + np.random.normal(0, 0.9, size=N)
)
loan_status = np.where(score > np.percentile(score, 31), "Y", "N")

df = pd.DataFrame({
    "Loan_ID": [f"LP{str(i).zfill(6)}" for i in range(1, N + 1)],
    "Gender": gender,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_amount_term,
    "Credit_History": credit_history,
    "Property_Area": property_area,
    "Loan_Status": loan_status,
})

df.to_csv("/home/claude/smart_lender/data/loan_data.csv", index=False)
print("Dataset created:", df.shape)
print(df["Loan_Status"].value_counts())
