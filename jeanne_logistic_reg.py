import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


pd.set_option('display.max_columns', None)
df = pd.read_csv('BreastCancer.csv')
cat_col = [col for col in df.columns if df[col].dtype == 'object']
num_col = [col for col in df.columns if df[col].dtype != 'object']
print('Categorical columns:', cat_col)
print('Numerical columns:', num_col)
round((df.isnull().sum() / df.shape[0]) * 100, 2)

sns.boxplot(x='diagnosis', y='radius_mean', data=df)
plt.title('Radius Mean - Benign vs Malignant')
plt.show()

df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

corr = df.corr()['diagnosis'].sort_values(ascending=False)
print(corr.head(11))


print("X shape:", X.shape)
print("y shape:", y.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Scaled:")
print(X_scaled[0])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc)

model = LogisticRegression(max_iter=10000, random_state=42)
model.fit(X_train, y_train)

print("Number of iterations:", model.n_iter_)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

print("First 10 predictions:", y_pred[:10])
print("First 10 actual values:", y_test.values[:10])

df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
numeric_cols = numeric_cols.drop('id')
df[numeric_cols] = df[numeric_cols] / df[numeric_cols].max()


X = df[['radius_mean', 'area_mean']]
y = df['diagnosis']

models = ['KNN', 'SVM', 'Decision Tree', 'Logistic Regression']
accuracy = [0.95, 0.97, 0.91, 0.97]
precision = [0.95, 0.97, 0.91, 0.97]
recall = [0.95, 0.96, 0.91, 0.97]

bar_width = 0.25
index = np.arange(len(models))


plt.figure(figsize=(12, 6))

plt.bar(index, accuracy, bar_width, label='Accuracy', color='#1f77b4')
plt.bar(index + bar_width, precision, bar_width, label='Precision', color='#ff7f0e')
plt.bar(index + 2 * bar_width, recall, bar_width, label='Recall', color='#2ca02c')


plt.title('Performance Metrics Comparison by Model', fontsize=16)
plt.xlabel('Models', fontsize=12)
plt.ylabel('Scores', fontsize=12)
plt.xticks(index + bar_width, models) 
plt.ylim(0, 1.0)
plt.legend(loc='lower right')


plt.show()





