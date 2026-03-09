import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

df = pd.read_csv(r"C:\Users\LENOVO\OneDrive\Documents\Sourcsyes\heart_disease\heart.csv")
X = df.drop("target", axis=1).values
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Find best k to fix overfitting
param_grid = {"n_neighbors": list(range(5, 31))}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="accuracy")
grid.fit(X_train_scaled, y_train)
best_k = grid.best_params_["n_neighbors"]

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  KNeighborsClassifier(n_neighbors=best_k))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracy_scores  = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
f1_scores        = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_weighted")
precision_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="precision_weighted")

print("=" * 55)
print("       5-FOLD CROSS VALIDATION RESULTS")
print("       KNN | Heart Disease")
print("=" * 55)
print(f"  Best K (n_neighbors) : {best_k}")
print("-" * 55)
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