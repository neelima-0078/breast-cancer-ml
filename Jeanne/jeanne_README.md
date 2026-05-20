Jeanne — Logistic Regression + Model Comparison
About This File
This script contains my individual contribution
to the Breast Cancer ML team project.
What This Script Does

Loads and explores the dataset
Identifies categorical and numerical columns
Checks for null values
EDA — boxplot of radius mean by diagnosis, correlation with target
Encodes diagnosis (M → 1, B → 0)
Scales features using StandardScaler
Splits data 80% training / 20% testing
Trains Logistic Regression model (max_iter=10000)
Evaluates model performance
Compares KNN, SVM, Decision Tree, and Logistic Regression side by side

My Results

Model: Logistic Regression
Accuracy: 97%
Precision: 97%
Recall: 97%
F1 Score: 97%

Model Comparison Results
ModelAccuracyPrecisionRecallKNN95%95%95%SVM97%97%96%Decision Tree91%91%91%Logistic Regression97%97%97%
Why Logistic Regression Is A Good Model

Matches SVM at 97% accuracy with simpler interpretability
Highest precision and recall balance across all 4 models
Decision Tree underperforms at 91% — overfitting risk without tuning
Logistic Regression is well-suited for binary medical classification tasks

How To Run

Upload BreastCancer.csv to same folder as script
Run jeanne_logistic_reg.py in any Python environment
Charts will display automatically during execution
EOF
