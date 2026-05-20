# Neelima — Logistic Regression + Prediction Algorithm

## About This File
This notebook contains my individual contribution 
to the Breast Cancer ML team project.

## What This Notebook Does
1. Loads and explores the dataset
2. Cleans data (removes id, Unnamed:32, encodes M/B)
3. EDA — count plot, heatmap, boxplot, top features
4. Scales features using StandardScaler
5. Splits data 80% training / 20% testing
6. Trains Logistic Regression model
7. Evaluates model performance
8. Tests predictions on 4 different patients
9. Prediction algorithm with recommendations

## My Results
- Model: Logistic Regression
- Accuracy: 97.37%
- Recall (Malignant): 95%
- Precision: 98%
- F1 Score: 96%

## Patient Prediction Results
- Patient 1 Small Tumor    → BENIGN     0.0% cancer
- Patient 2 Medium Tumor   → BENIGN    47.1% cancer
- Patient 3 Large Tumor    → MALIGNANT 100.0% cancer
- Patient 4 Very Large     → MALIGNANT 100.0% cancer

## Why Logistic Regression Is Best Model
- Highest recall (95%) catches most cancer cases
- 97.37% accuracy tied with SVM
- No overfitting detected
- Most interpretable for medical diagnosis

## How To Run
1. Upload data.csv to same folder as notebook
2. Open neelimacode_breastcancer.ipynb in Jupyter
3. Run all cells from top to bottom
