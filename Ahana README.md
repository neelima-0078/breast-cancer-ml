# Breast Cancer Prediction using K-Nearest Neighbors (KNN)

## About This File

This notebook contains my individual contribution to the Breast Cancer ML team project.  
As part of the team project, each member worked on a different model. I selected K-Nearest Neighbors (KNN) to implement and evaluate.

---

## What This Notebook Does

1. Loads and cleans the breast cancer dataset  
2. Removes unnecessary columns (`id`, `Unnamed: 32`)  
3. Converts diagnosis values (`M` → 1, `B` → 0)  
4. Performs basic EDA and statistical analysis  
5. Creates distribution plots for all features  
6. Generates a correlation heatmap  
7. Splits data into training and testing sets  
8. Scales features using `StandardScaler`  
9. Trains a K-Nearest Neighbors (KNN) model  
10. Evaluates the model using accuracy, classification report, and confusion matrix  
11. Tests different K values from 1–20  
12. Saves the trained model and scaler using Pickle  

---

## Model Details

- **Model Used:** K-Nearest Neighbors (KNN)  
- **K Value:** 5  
- **Feature Scaling:** StandardScaler  
- **Train-Test Split:** 80% Training / 20% Testing  

---

## Visualizations Included

- Feature Distribution Histograms  
- Correlation Heatmap  
- Confusion Matrix  
- K vs Accuracy Graph  

---

## Evaluation Metrics Used

- Accuracy Score  
- Classification Report  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix  

---

## Strengths of KNN

- Simple and effective classification algorithm  
- Works well on numerical datasets  
- Easy to understand and implement  
- Performs better after feature scaling  
- Uses distance-based classification to identify nearby data points  

---


## How To Run

1. Upload `Cancer.csv` into the same folder as the notebook  
2. Open the notebook in Jupyter Notebook or VS Code  
3. Run all cells from top to bottom  

---

## My Results
- Model: K-Nearest Neighbors (KNN)
- K Value: 5
- Accuracy: ~96% (varies slightly based on run)
- Best K (from tuning): 5–7 range gave stable performance

---

## Prediction Insights
- Model successfully classified most cases with high accuracy
- Clear separation between Benign and Malignant cases after scaling
- Malignant cases were detected with high reliability due to distance-based learning

---

## Sample Prediction Behavior (Conceptual)
- Small feature values → Mostly BENIGN (0)
- Medium feature values → MIXED predictions depending on neighborhood
- High feature values → Mostly MALIGNANT (1)
