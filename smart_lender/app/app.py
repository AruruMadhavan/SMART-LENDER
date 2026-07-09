"""
SMART LENDER - Flask Web Application (Epic 5: Application Building)
======================================================================
Loads the best model saved by notebook/smart_lender_pipeline.py and serves
a simple form. User fills applicant details -> app predicts
"Loan Approved" / "Loan Rejected".

Run:
    cd smart_lender/app
    python3 app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import pickle
import json
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_DIR = "../models"

with open(f"{MODEL_DIR}/best_model.pkl", "rb") as f:
    model = pickle.load(f)
with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open(f"{MODEL_DIR}/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)
with open(f"{MODEL_DIR}/feature_columns.json") as f:
    feature_columns = json.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form

    gender = encoders["Gender"].transform([form["gender"]])[0]
    married = encoders["Married"].transform([form["married"]])[0]
    dependents = encoders["Dependents"].transform([form["dependents"]])[0]
    education = encoders["Education"].transform([form["education"]])[0]
    self_employed = encoders["Self_Employed"].transform([form["self_employed"]])[0]
    property_area = encoders["Property_Area"].transform([form["property_area"]])[0]

    loan_amount_term = float(form["loan_amount_term"])
    credit_history = float(form["credit_history"])

    applicant_income = float(form["applicant_income"])
    coapplicant_income = float(form["coapplicant_income"])
    loan_amount = float(form["loan_amount"])

    total_income_log = np.log1p(applicant_income + coapplicant_income)
    loan_amount_log = np.log1p(loan_amount)

    row = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
        "LoanAmount_log": loan_amount_log,
        "TotalIncome_log": total_income_log,
    }
    X = np.array([[row[c] for c in feature_columns]])
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0][1] if hasattr(model, "predict_proba") else None

    result = "Loan Approved ✅" if pred == 1 else "Loan Rejected ❌"
    confidence = f"{proba*100:.1f}%" if proba is not None else "N/A"

    return render_template("result.html", result=result, confidence=confidence)


if __name__ == "__main__":
    app.run(debug=True)
