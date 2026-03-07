import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

df = pd.read_csv(r"C:\Users\LENOVO\OneDrive\Documents\Sourcsyes\heart_disease\heart.csv")
X = df.drop("target", axis=1).values
y = df["target"].values

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  LogisticRegression(max_iter=1000, random_state=42))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracy_scores  = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
f1_scores        = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_weighted")
precision_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="precision_weighted")

print("=" * 55)
print("       5-FOLD CROSS VALIDATION RESULTS")
print("       Logistic Regression | Heart Disease")
print("=" * 55)
print(f"  {'Fold':<8} {'Accuracy':>12} {'F1 Score':>12} {'Precision':>12}")
print("-" * 55)
for i in range(5):
    print(f"  Fold {i+1:<3}  {accuracy_scores[i]:>12.4f} {f1_scores[i]:>12.4f} {precision_scores[i]:>12.4f}")
print("-" * 55)
print(f"  {'Mean':<8} {accuracy_scores.mean():>12.4f} {f1_scores.mean():>12.4f} {precision_scores.mean():>12.4f}")
print(f"  {'Std':<8} {accuracy_scores.std():>12.4f} {f1_scores.std():>12.4f} {precision_scores.std():>12.4f}")
print("=" * 55)
print()
print(f"  Mean Accuracy  : {accuracy_scores.mean() * 100:.2f}% ± {accuracy_scores.std() * 100:.2f}%")
print(f"  Mean F1 Score  : {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
print(f"  Mean Precision : {precision_scores.mean():.4f} ± {precision_scores.std():.4f}")
print("=" * 55)