# ============================================
# BREAST CANCER DIAGNOSIS - FINAL PROJECT
# Model: Logistic Regression
# ============================================

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

warnings.filterwarnings("ignore")


# ============================================
# 1. CONFIG
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data.csv")
MODEL_FILE = os.path.join(BASE_DIR, "logistic_regression_model.pkl")
SCALER_FILE = os.path.join(BASE_DIR, "scaler.pkl")
RANDOM_STATE = 42


# ============================================
# 2. LOAD AND CLEAN DATA
# ============================================

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)

    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")

    if "diagnosis" not in df.columns:
        raise ValueError("Target column 'diagnosis' not found in dataset.")

    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    if df["diagnosis"].isnull().any():
        raise ValueError("Diagnosis column contains unexpected values.")

    return df


# ============================================
# 3. BASIC EDA
# ============================================

def show_eda(df):
    print("\nData Loaded Successfully")
    print(df.head())

    print("\nDataset Info:")
    print(df.info())

    print("\nSummary Statistics:")
    print(df.describe())

    print("\nMissing Values:")
    print(df.isnull().sum())

    plt.figure(figsize=(6, 4))
    sns.countplot(x="diagnosis", data=df, palette=["steelblue", "salmon"])
    plt.title("Benign vs Malignant Count")
    plt.xticks([0, 1], ["Benign (0)", "Malignant (1)"])
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.boxplot(x="diagnosis", y="radius_mean", data=df, palette=["steelblue", "salmon"])
    plt.title("Radius Mean by Diagnosis")
    plt.xticks([0, 1], ["Benign (0)", "Malignant (1)"])
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 10))
    sns.heatmap(df.corr(), cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    corr = df.corr()["diagnosis"].sort_values(ascending=False)
    print("\nTop features correlated with diagnosis:")
    print(corr.head(11))


# ============================================
# 4. TRAIN-TEST SPLIT AND SCALING
# ============================================

def prepare_data(df):
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nData Split Completed")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    print("\nScaling Done")
    print("X_train_scaled:", X_train_scaled.shape)
    print("X_test_scaled :", X_test_scaled.shape)

    return X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


# ============================================
# 5. LOGISTIC REGRESSION MODEL
# ============================================

def train_model(X_train_scaled, y_train):
    model = LogisticRegression(max_iter=10000, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    print("\nLogistic Regression Model Trained")
    print("Number of iterations:", model.n_iter_)

    return model


# ============================================
# 6. MODEL EVALUATION
# ============================================

def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    print("\nModel Evaluation")
    print("----------------")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Benign (0)", "Malignant (1)"]
    ))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred Benign", "Pred Malignant"],
        yticklabels=["Actual Benign", "Actual Malignant"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Logistic Regression")
    plt.tight_layout()
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f"ROC curve (AUC = {roc_auc:.3f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Logistic Regression")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    print(f"\nROC-AUC Score: {roc_auc:.4f}")

    return y_pred, y_prob, roc_auc


# ============================================
# 7. SAVE MODEL AND SCALER
# ============================================

def save_assets(model, scaler):
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nModel saved as: {MODEL_FILE}")
    print(f"Scaler saved as: {SCALER_FILE}")


# ============================================
# 8. PREDICTION FUNCTION WITH RECOMMENDATIONS
# ============================================

def predict_patient(input_values, model, scaler, feature_names, patient_name="Patient"):
    if len(input_values) != len(feature_names):
        raise ValueError(f"Expected {len(feature_names)} input values, got {len(input_values)}.")

    input_df = pd.DataFrame([input_values], columns=feature_names)
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    cancer_prob = probability[1] * 100
    benign_prob = probability[0] * 100

    print("\n" + "=" * 55)
    print("    🎗️  BREAST CANCER PREDICTION SYSTEM")
    print("=" * 55)
    print(f"\nPatient Name: {patient_name}")
    print(f"\n📊 Probability Scores:")
    print(f"   Cancer (Malignant): {cancer_prob:.1f}%")
    print(f"   Benign:             {benign_prob:.1f}%")

    if prediction == 1:
        print(f"\n🔴 DIAGNOSIS: MALIGNANT")
        print(f"\n📋 Recommendation:")
        if cancer_prob >= 90:
            print("   ⚠️  HIGH RISK")
            print("   → See an oncologist immediately")
            print("   → Biopsy strongly recommended")
            print("   → Do not delay treatment")
        elif cancer_prob >= 70:
            print("   ⚠️  MODERATE RISK")
            print("   → Book appointment this week")
            print("   → Further imaging tests needed")
        else:
            print("   ⚠️  BORDERLINE")
            print("   → Get a second medical opinion")
            print("   → Follow up scan in 1 month")
    else:
        print(f"\n🟢 DIAGNOSIS: BENIGN")
        print(f"\n📋 Recommendation:")
        if benign_prob >= 90:
            print("   ✅ LOW RISK")
            print("   → Continue regular annual check-ups")
            print("   → Maintain healthy lifestyle")
        else:
            print("   ✅ LIKELY BENIGN")
            print("   → Follow up in 3 months")
            print("   → Monitor for any changes")

    print(f"\n⚠️  This tool supports diagnosis only.")
    print("   Always consult a qualified doctor.")
    print("=" * 55)

    return {
        "patient_name": patient_name,
        "prediction": int(prediction),
        "diagnosis": "Malignant" if prediction == 1 else "Benign",
        "cancer_probability": cancer_prob,
        "benign_probability": benign_prob
    }


# ============================================
# 9. SAMPLE TEST CASES
# ============================================

def run_sample_predictions(model, scaler, feature_names):
    patients = {
        "Patient 1 - Small Tumor": [
            10.0, 14.0, 65.0, 300.0, 0.08, 0.05, 0.02, 0.01,
            0.17, 0.06, 0.2, 0.8, 1.5, 15.0, 0.004, 0.01,
            0.01, 0.003, 0.015, 0.003, 11.0, 18.0, 72.0,
            365.0, 0.10, 0.08, 0.05, 0.02, 0.25, 0.07
        ],
        "Patient 2 - Medium Tumor": [
            14.0, 20.0, 95.0, 600.0, 0.10, 0.12, 0.10, 0.06,
            0.19, 0.07, 0.5, 1.5, 3.0, 45.0, 0.007, 0.025,
            0.03, 0.008, 0.02, 0.004, 16.0, 26.0, 108.0,
            800.0, 0.13, 0.20, 0.22, 0.09, 0.31, 0.09
        ],
        "Patient 3 - Large Tumor": [
            20.0, 25.0, 135.0, 1200.0, 0.14, 0.25, 0.28, 0.16,
            0.22, 0.09, 0.9, 2.5, 6.0, 90.0, 0.013, 0.05,
            0.07, 0.018, 0.035, 0.007, 26.0, 35.0, 175.0,
            2000.0, 0.18, 0.40, 0.50, 0.22, 0.40, 0.12
        ],
        "Patient 4 - Very Large Tumor": [
            25.0, 30.0, 165.0, 2000.0, 0.16, 0.35, 0.40, 0.20,
            0.25, 0.10, 1.2, 3.0, 8.0, 120.0, 0.015, 0.07,
            0.09, 0.022, 0.04, 0.009, 30.0, 40.0, 200.0,
            2500.0, 0.20, 0.55, 0.65, 0.28, 0.45, 0.14
        ]
    }

    print("\n" + "=" * 55)
    print("  CANCER OUTPUT FOR DIFFERENT PATIENT INPUTS")
    print("=" * 55)

    for name, values in patients.items():
        predict_patient(values, model, scaler, feature_names, patient_name=name)


# ============================================
# 10. OPTIONAL MANUAL INPUT
# ============================================

def manual_prediction(model, scaler, feature_names):
    choice = input("\nDo you want to enter your own patient values? (y/n): ").strip().lower()
    if choice not in ["y", "yes"]:
        return

    print("\nEnter values for all features below:")
    user_values = []

    for feature in feature_names:
        while True:
            try:
                value = float(input(f"{feature}: "))
                user_values.append(value)
                break
            except ValueError:
                print("Please enter a valid numeric value.")

    patient_name = input("\nEnter patient name: ").strip() or "Manual Patient"
    predict_patient(user_values, model, scaler, feature_names, patient_name=patient_name)


# ============================================
# 11. MAIN
# ============================================

def main():
    if not os.path.exists(DATA_FILE):
        print(f"Error: '{DATA_FILE}' not found.")
        return

    df = load_and_clean_data(DATA_FILE)
    show_eda(df)

    X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = prepare_data(df)

    model = train_model(X_train_scaled, y_train)

    evaluate_model(model, X_test_scaled, y_test)

    save_assets(model, scaler)

    run_sample_predictions(model, scaler, X.columns.tolist())

    manual_prediction(model, scaler, X.columns.tolist())


if __name__ == "__main__":
    main()