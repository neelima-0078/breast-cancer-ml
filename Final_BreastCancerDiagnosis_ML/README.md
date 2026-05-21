# Breast Cancer Diagnosis using Logistic Regression

## Project Overview
This project is a machine learning-based breast cancer diagnosis system built using **Logistic Regression**.  
It is designed to classify tumor cases into **Benign** or **Malignant** using the Breast Cancer Wisconsin dataset.

The final script combines data preprocessing, exploratory data analysis, model training, evaluation, prediction, and recommendation logic into one runnable Python file.

## Project Objective
The objective of this project is to build a binary classification system that can:
- Analyze tumor-related medical features
- Predict whether a tumor is benign or malignant
- Provide probability-based output
- Display recommendation messages based on the prediction result

## Project Type
- **Domain:** Healthcare / Medical Diagnosis
- **Task:** Binary Classification
- **Model Used:** Logistic Regression

## Dataset Information
The dataset used in this project contains measurements computed from digitized images of breast mass cell nuclei.

### Target Variable
- `diagnosis`
  - `M` = Malignant
  - `B` = Benign

### Data Cleaning Performed
The following preprocessing steps are applied in the final script:
- Removed the `id` column because it does not help prediction
- Removed the `Unnamed: 32` column because it is empty/unnecessary
- Encoded diagnosis values:
  - `M → 1`
  - `B → 0`

## Features Used
The dataset contains 30 numerical features related to tumor properties, including:
- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Symmetry
- Fractal dimension

These measurements are used by the model to learn patterns associated with benign and malignant tumors.

## Final Script Contents
The file `finalscript.py` includes the complete final workflow of the project:

1. Importing required libraries
2. Loading the dataset
3. Cleaning and preprocessing the data
4. Performing exploratory data analysis (EDA)
5. Splitting the dataset into training and testing sets
6. Scaling the features using `StandardScaler`
7. Training the Logistic Regression model
8. Evaluating performance using standard classification metrics
9. Plotting confusion matrix and ROC curve
10. Saving the trained model and scaler
11. Running predictions on sample patient inputs
12. Allowing manual patient input for new predictions

## Exploratory Data Analysis
The script includes visualizations to better understand the dataset before training:

- **Count Plot**  
  Shows the number of benign and malignant cases

- **Box Plot**  
  Compares `radius_mean` across diagnosis classes

- **Correlation Heatmap**  
  Shows relationships between features and target label

- **Top Correlated Features**  
  Displays the features most strongly associated with diagnosis

## Data Splitting and Scaling
The final script uses:
- **80% training data**
- **20% testing data**

The split is performed using `train_test_split()` with:
- `random_state = 42`
- `stratify = y`

Feature scaling is done using `StandardScaler`.

### Important Note
The scaler is fitted only on the training data and then applied to the test data.  
This is important because it avoids **data leakage** and makes the evaluation more valid.

## Model Training
The final model used in the project is:

- **Logistic Regression**
- `max_iter = 10000`
- `random_state = 42`

Logistic Regression was selected because:
- It is well suited for binary classification
- It provides class probability scores
- It is interpretable compared to more complex models
- It gave strong performance for this dataset

## Model Evaluation
The script evaluates the model using the following metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **ROC-AUC Score**

It also generates:
- **Classification Report**
- **Confusion Matrix**
- **ROC Curve**

These metrics are printed automatically when the script runs.

## Prediction Algorithm
One of the main parts of the final project is the prediction algorithm.

The algorithm:
- Accepts patient feature values
- Converts them into a dataframe
- Scales them using the fitted scaler
- Predicts the class using the trained Logistic Regression model
- Calculates:
  - Malignant probability
  - Benign probability
- Displays diagnosis and recommendation output

## Recommendation System
The prediction system includes recommendation logic inspired by the team’s final approach.

### If the prediction is Malignant
- **High Risk**
  - See an oncologist immediately
  - Biopsy strongly recommended
  - Do not delay treatment

- **Moderate Risk**
  - Book appointment this week
  - Further imaging tests needed

- **Borderline**
  - Get a second medical opinion
  - Follow up scan in 1 month

### If the prediction is Benign
- **Low Risk**
  - Continue regular annual check-ups
  - Maintain healthy lifestyle

- **Likely Benign**
  - Follow up in 3 months
  - Monitor for any changes

## Output Files
After running the script, the following files are created:

- `logistic_regression_model.pkl`  
  Saved trained Logistic Regression model

- `scaler.pkl`  
  Saved fitted StandardScaler object

These files can be reused later without retraining the model.

## Required Files
To run the project, keep these files in the same folder:

- `finalscript.py`
- `data.csv`

## Requirements
The script uses the following Python libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `pickle`
- `os`

If needed, install the required packages using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How To Run
1. Place `data.csv` in the same folder as `finalscript.py`
2. Open Terminal or Command Prompt
3. Navigate to the folder containing the files
4. Run the script:

```bash
python3 finalscript.py
```

## What Happens When the Script Runs
When executed, the script will:

1. Load and clean the dataset
2. Display dataset information and summary statistics
3. Show EDA plots
4. Split and scale the data
5. Train the Logistic Regression model
6. Print model evaluation results
7. Display confusion matrix and ROC curve
8. Save the trained model and scaler
9. Run prediction tests on sample patient inputs
10. Ask whether the user wants to enter manual patient values

## Sample Prediction Cases
The script includes predefined patient examples to demonstrate how the prediction algorithm works.  
These sample cases simulate tumors of different sizes and feature values so that the model output can be tested directly.

## Why This Final Script Is Important
This final script represents the complete version of the team project because it combines:
- Data preprocessing
- Visualization
- Model training
- Model evaluation
- Prediction algorithm
- Recommendation logic

Instead of keeping separate model files from each member, this script keeps only the components needed for the final runnable project.

## Limitations
Although the project performs prediction using machine learning, it has limitations:

- It is based only on the given dataset
- It should not be used as a substitute for professional diagnosis
- Real medical decisions must always be made by qualified doctors

## Conclusion
This project demonstrates how Logistic Regression can be used for breast cancer diagnosis as a binary classification problem.  
It shows the full machine learning pipeline from raw dataset to prediction and recommendation output in one final script.

## Important Note
This tool is created for **educational and project purposes only**.  
It does **not** replace medical testing, biopsy, imaging, or professional consultation.  
Always consult a qualified doctor for any medical diagnosis.
