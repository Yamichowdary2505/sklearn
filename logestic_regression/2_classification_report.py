import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

data = load_breast_cancer()
X, y = data.data, data.target
target_names = data.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("=" * 50)
print("        CONFUSION MATRIX")
print("        Logistic Regression | Breast Cancer")
print("=" * 50)
print(f"                    Predicted")
print(f"                    Malignant   Benign")
print(f"  Actual Malignant    {cm[0][0]:>5}      {cm[0][1]:>5}")
print(f"  Actual Benign       {cm[1][0]:>5}      {cm[1][1]:>5}")
print("=" * 50)

print("\n" + "=" * 50)
print("   PRECISION, RECALL, F1, SUPPORT")
print("=" * 50)
print(classification_report(y_test, y_pred, target_names=target_names))
print("=" * 50)
