import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("/Users/haniafaizal/Desktop/jump/breastcancer proj/data.csv")

if "Unnamed: 32" in df.columns:
    df = df.drop(columns=["Unnamed: 32"])

if "id" in df.columns:
    df = df.drop(columns=["id"])

df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

print("Features and target created successfully.")
print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain-test split completed successfully.")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nStandard scaling completed successfully.")
print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)




#svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Check shapes (for documentation)
print("X_train_scaled:", X_train_scaled.shape)
print("X_test_scaled :", X_test_scaled.shape)
print("y_train       :", y_train.shape)
print("y_test        :", y_test.shape)

# 2) Create and train SVM with RBF kernel
svm_rbf = SVC(kernel="rbf", probability=True, random_state=42)
svm_rbf.fit(X_train_scaled, y_train)
print("\nSVM (RBF kernel) training completed.")

# 3) Predict on test set
y_pred_svm = svm_rbf.predict(X_test_scaled)
print("Predictions on test data generated.")

# 4) Basic evaluation
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("\nSVM Accuracy on test set:", round(accuracy_svm, 4))

print("\nClassification Report (SVM):")
print(classification_report(y_test, y_pred_svm, target_names=["Benign (0)", "Malignant (1)"]))

# 5) Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred Benign", "Pred Malignant"],
            yticklabels=["Actual Benign", "Actual Malignant"])
plt.title("SVM Confusion Matrix (Test Set)")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.show()

# 6) ROC curve and AUC
y_scores_svm = svm_rbf.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_scores_svm)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="darkorange", lw=2,
         label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("SVM ROC Curve (Malignant vs Benign)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

print("\nROC-AUC for SVM:", round(roc_auc, 4))


#line graph
import matplotlib.pyplot as plt

models = ["Logistic Regression", "SVM", "KNN", "Decision Tree"]

accuracy = [97.37, 97.37, 94.74, 98.24]
f1_benign = [0.98, 0.98, 0.96, 0.93]
f1_malignant = [0.96, 0.96, 0.93, 0.87]

# Combined score
combined_score = [
    (a + b*100 + m*100) / 3
    for a, b, m in zip(accuracy, f1_benign, f1_malignant)
]

plt.figure(figsize=(9, 5))
plt.plot(models, combined_score, marker='o', linewidth=2.5, color='darkcyan')

for i, value in enumerate(combined_score):
    plt.text(i, value + 0.2, f"{value:.2f}", ha='center', fontsize=10)

plt.title("Combined Model Performance Score")
plt.xlabel("Models")
plt.ylabel("Balanced Performance Score")
plt.ylim(91, 98)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()


#checking output for diff inputs
# checking output for diff inputs
import pandas as pd

def get_recommendation(prob):
    if prob < 0.30:
        return "Low risk – likely benign, but medical confirmation is advised."
    elif prob < 0.70:
        return "Moderate risk – further medical review is recommended."
    else:
        return "High risk – urgent clinical evaluation is strongly recommended."

sample_indices = [0, 5, 10, 20, 30]

selected_inputs = X_test.iloc[sample_indices]
selected_inputs_scaled = scaler.transform(selected_inputs)

# USE THE TRAINED SVM MODEL HERE
pred_labels = svm_rbf.predict(selected_inputs_scaled)
pred_probs = svm_rbf.predict_proba(selected_inputs_scaled)[:, 1]

results = pd.DataFrame({
    "Patient Index": selected_inputs.index,
    "Predicted Diagnosis": ["Malignant" if p == 1 else "Benign" for p in pred_labels],
    "Cancer Probability": [round(prob, 4) for prob in pred_probs],
    "Recommendation": [get_recommendation(prob) for prob in pred_probs]
})

print(results)


#predication algo supple
def predict_patient(features_row, patient_name="Patient"):
    scaled = scaler.transform(features_row)
    prob = svm_rbf.predict_proba(scaled)[:, 1][0]
    label = svm_rbf.predict(scaled)[0]

    diagnosis = "Malignant" if label == 1 else "Benign"

    if prob < 0.30:
        risk_level = "Low Risk"
        recommendation = "Likely benign, but medical confirmation is advised."
    elif prob < 0.70:
        risk_level = "Moderate Risk"
        recommendation = "Further medical review is recommended."
    else:
        risk_level = "High Risk"
        recommendation = "Urgent clinical evaluation is strongly recommended."

    print("\n--- Prediction Result ---")
    print(f"Patient Name         : {patient_name}")
    print(f"Predicted Diagnosis  : {diagnosis}")
    print(f"Cancer Probability   : {prob*100:.2f}%")
    print(f"Risk Level           : {risk_level}")
    print(f"Recommendation       : {recommendation}")
    print("-------------------------")

example_patient = X_test.iloc[[0]]
predict_patient(example_patient, "Sample Patient 1")

#tweaked
# Base patient (same as before)
base_patient = X_test.iloc[[0]]

# Original prediction
predict_patient(base_patient, "Original Patient")

# Tweak 1: decrease radius_mean
predict_patient(
    base_patient.assign(radius_mean=base_patient["radius_mean"] - 2),
    "radius_mean decreased by 2"
)

# Tweak 2: increase radius_mean
predict_patient(
    base_patient.assign(radius_mean=base_patient["radius_mean"] + 2),
    "radius_mean increased by 2"
)