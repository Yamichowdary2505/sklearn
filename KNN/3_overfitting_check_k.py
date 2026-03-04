import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc  = accuracy_score(y_test,  model.predict(X_test))
gap       = train_acc - test_acc

if gap > 0.10:
    status = "OVERFITTING DETECTED"
    detail = "Model performs much better on training data than test data."
elif test_acc < 0.75:
    status = "UNDERFITTING DETECTED"
    detail = "Model performs poorly on both training and test data."
else:
    status = "MODEL IS HEALTHY"
    detail = "Train and test accuracy are close — no overfitting or underfitting."

print("=" * 55)
print("       OVERFITTING / UNDERFITTING CHECK")
print("       KNN | Breast Cancer")
print("=" * 55)
print(f"  Train Accuracy   : {train_acc * 100:.2f}%")
print(f"  Test  Accuracy   : {test_acc  * 100:.2f}%")
print(f"  Gap              : {gap * 100:.2f}%")
print("-" * 55)
print(f"  Status           : {status}")
print(f"  Detail           : {detail}")
print("=" * 55)
print()
print("  Reference Guide:")
print("  Gap < 5%   → No overfitting  ✅")
print("  Gap 5-10%  → Slight overfitting ⚠️")
print("  Gap > 10%  → Overfitting ❌")
print("  Test < 75% → Underfitting ❌")
print("=" * 55)
