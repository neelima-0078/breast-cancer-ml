Hania - SVM + Prediction Algorithm
About This File
This notebook contains my individual contribution to the Breast Cancer ML team project.
What This Notebook Does

Loads the dataset from CSV
Cleans data (removes id, Unnamed: 32, encodes M/B)
Splits data 80% training / 20% testing with stratification
Scales features using StandardScaler
Trains an SVM model with RBF kernel
Evaluates model performance (accuracy, classification report, confusion matrix)
Plots ROC curve and calculates AUC score
Compares SVM against other models (Logistic Regression, KNN, Decision Tree)
Tests predictions on 5 sample patients with risk recommendations
Prediction function with 3-tier risk level output

My Results

Model: SVM (RBF kernel)
Accuracy: 97.37%
ROC-AUC: ~0.997
F1 Score (Benign): 98%
F1 Score (Malignant): 96%

Patient Prediction Results

Patient Index 0 → Benign — Low risk
Patient Index 5 → Benign — Low risk
Patient Index 10 → Malignant — High risk
Patient Index 20 → Malignant — High risk
Patient Index 30 → Benign — Moderate risk

Risk Recommendation Logic

< 30% probability → Low risk: likely benign, medical confirmation advised
30–70% probability → Moderate risk: further medical review recommended


70% probability → High risk: urgent clinical evaluation strongly recommended



Why SVM Is a Strong Model

Tied highest accuracy (97.37%) with Logistic Regression
Excellent AUC (~0.997) — near-perfect class separation
RBF kernel handles non-linear decision boundaries well
Robust performance on high-dimensional medical data

How To Run

Upload data.csv to the same folder as the notebook
Open hania_SVM.ipynb in Jupyter or Google Colab
Run all cells from top to bottom
