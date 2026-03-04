import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, mean_squared_error, r2_score
)

data = load_breast_cancer()
X, y = data.data, data.target
target_names = data.target_names

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  KNeighborsClassifier(n_neighbors=5))
])

cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(pipeline, X, y, cv=cv)

accuracy = accuracy_score(y, y_pred)
mse      = mean_squared_error(y, y_pred)
r2       = r2_score(y, y_pred)
cm       = confusion_matrix(y, y_pred)
report   = classification_report(y, y_pred, target_names=target_names)

print("=" * 55)
print("       CROSS VALIDATE PREDICT RESULTS")
print("       KNN | Breast Cancer")
print("       5-Fold Stratified CV")
print("=" * 55)
print(f"  Total Samples    : {len(y)}")
print(f"  Folds            : 5")
print("-" * 55)
print(f"  Overall Accuracy : {accuracy * 100:.2f}%")
print(f"  MSE              : {mse:.4f}")
print(f"  R² Score         : {r2:.4f}")
print("=" * 55)

print("\n" + "=" * 55)
print("       CONFUSION MATRIX (Cross Val Predict)")
print("=" * 55)
print(f"                    Predicted")
print(f"                    Malignant   Benign")
print(f"  Actual Malignant    {cm[0][0]:>5}      {cm[0][1]:>5}")
print(f"  Actual Benign       {cm[1][0]:>5}      {cm[1][1]:>5}")
print("=" * 55)

print("\n" + "=" * 55)
print("       CLASSIFICATION REPORT (Cross Val Predict)")
print("=" * 55)
print(report)
print("=" * 55)

print("\n" + "=" * 55)
print("       SAMPLE PREDICTIONS (First 20)")
print("=" * 55)
print(f"  {'Sample':<10} {'Actual':<15} {'Predicted':<15} {'Correct'}")
print("-" * 55)
for i in range(20):
    actual    = target_names[y[i]]
    predicted = target_names[y_pred[i]]
    correct   = "✅" if y[i] == y_pred[i] else "❌"
    print(f"  {i+1:<10} {actual:<15} {predicted:<15} {correct}")
print("=" * 55)
