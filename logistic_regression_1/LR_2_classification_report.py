import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv(r"C:\Users\LENOVO\OneDrive\Documents\Sourcsyes\heart_disease\heart.csv")
X = df.drop("target", axis=1).values
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
cm     = confusion_matrix(y_test, y_pred)

print("=" * 50)
print("        CONFUSION MATRIX")
print("        Logistic Regression | Heart Disease")
print("=" * 50)
print(f"                    Predicted")
print(f"                    No Disease  Has Disease")
print(f"  Actual No Disease    {cm[0][0]:>5}      {cm[0][1]:>5}")
print(f"  Actual Has Disease   {cm[1][0]:>5}      {cm[1][1]:>5}")
print("=" * 50)
print("\n" + "=" * 50)
print("   PRECISION, RECALL, F1, SUPPORT")
print("=" * 50)
print(classification_report(y_test, y_pred, target_names=["No Disease", "Has Disease"]))
print("=" * 50)