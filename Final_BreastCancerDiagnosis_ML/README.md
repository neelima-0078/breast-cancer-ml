# Final Script — Logistic Regression + Prediction Algorithm

## About This File
This Python script contains the final combined version  
of our Breast Cancer ML team project.

## What This Script Does
1. Loads and explores the dataset
2. Cleans data (removes `id`, `Unnamed: 32`, encodes `M/B`)
3. Performs EDA — count plot, heatmap, boxplot, top correlated features
4. Splits data into 80% training / 20% testing
5. Scales features using `StandardScaler`
6. Trains the Logistic Regression model
7. Evaluates model performance
8. Displays confusion matrix and ROC curve
9. Saves the trained model and scaler
10. Tests predictions on sample patients
11. Runs a prediction algorithm with recommendations

## Final Model Used
- Model: Logistic Regression
- Type: Binary Classification
- Target Classes:
  - `0` = Benign
  - `1` = Malignant

## Evaluation Metrics
- Accuracy: Based on test set results from the final script
- Precision: Shown in the classification report
- Recall: Shown in the classification report
- F1 Score: Shown in the classification report
- ROC-AUC: Calculated from the ROC curve

## Prediction Algorithm
The final prediction algorithm:
- Accepts patient feature values
- Scales the input using the saved scaler
- Predicts whether the tumor is benign or malignant
- Shows malignant and benign probability scores
- Prints recommendation messages based on risk level

## Recommendation Logic
### If prediction is Malignant
- **High Risk** → See an oncologist immediately, biopsy strongly recommended, do not delay treatment
- **Moderate Risk** → Book appointment this week, further imaging tests needed
- **Borderline** → Get a second medical opinion, follow-up scan in 1 month

### If prediction is Benign
- **Low Risk** → Continue regular annual check-ups, maintain healthy lifestyle
- **Likely Benign** → Follow up in 3 months, monitor for any changes

## Why Logistic Regression Was Chosen
- Best suited for binary classification
- Gives probability output using `predict_proba()`
- Easy to interpret compared to more complex models
- Strong performance for this dataset
- Good fit for the final diagnosis + recommendation system

## Files Generated
- `logistic_regression_model.pkl` → saved trained model
- `scaler.pkl` → saved fitted scaler

## How To Run
1. Keep `finalscript.py` and `data.csv` in the same folder
2. Open Terminal
3. Move to the project folder:
   ```bash
   cd "/Users/haniafaizal/Desktop/jump/breastcancer proj"
   ```
4. Run the script:
   ```bash
   python3 finalscript.py
   ```

## Important Note
This tool is for educational and project purposes only.  
It supports prediction using machine learning, but it does **not** replace professional medical diagnosis.  
Always consult a qualified doctor.
