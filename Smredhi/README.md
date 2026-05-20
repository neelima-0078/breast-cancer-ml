#Smredhi — Decision Tree + Prediction Interface
About This File
This notebook contains my individual contribution
to the Breast Cancer ML team project.
What This Notebook Does

Loads and explores the dataset
Cleans data (removes id, Unnamed:32, encodes M/B)
EDA — bar chart, correlation heatmap, scatter plot, boxplot, top features
Scales features using StandardScaler (+ RobustScaler for comparison)
Splits data 80% training / 20% testing
Trains Decision Tree model
Evaluates model performance
Visualises the tree and feature importances
Saves model and scaler to .pkl files
Interactive CLI prediction interface with patient report

My Results

Model: Decision Tree Classifier
Training Accuracy: see notebook output
Test Accuracy: see notebook output
AUC-ROC: see notebook output
Precision / Recall / F1: see classification report

Patient Prediction Results (CLI Interface)

Enter any patient name, age, and 30 feature values
Outputs predicted diagnosis (Benign / Malignant)
Shows confidence %, probability breakdown, top 3 contributing features

Why Decision Tree Is A Good Model

Fully interpretable — tree structure can be read and explained
No black-box decisions, suitable for medical use cases
Feature importance scores show exactly which measurements drive predictions
max_depth=5 prevents overfitting while keeping the tree readable

How To Run

Upload data.csv to same folder as notebook
Open smredhi_decisionTree_ml.ipynb in Jupyter
Run all cells from top to bottom
Saved files decision_tree_model.pkl and scaler.pkl will be generated automatically
