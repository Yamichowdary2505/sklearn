import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names
target_names  = data.target_names

print(f"Total Samples   : {X.shape[0]}")
print(f"Total Features  : {X.shape[1]}")
print(f"Classes         : {list(target_names)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train Samples   : {X_train.shape[0]}")
print(f"Test  Samples   : {X_test.shape[0]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print(f"\nStandard Scaler:")
print(f"  Mean before scaling : {X_train[:, 0].mean():.4f}")
print(f"  Mean after  scaling : {X_train_scaled[:, 0].mean():.4f}")
print(f"  Std  before scaling : {X_train[:, 0].std():.4f}")
print(f"  Std  after  scaling : {X_train_scaled[:, 0].std():.4f}")

X_with_missing = X_train_scaled.copy()
np.random.seed(42)
missing_indices = np.random.choice(X_with_missing.size, size=20, replace=False)
X_with_missing.flat[missing_indices] = np.nan

imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X_with_missing)
print(f"\nSimple Imputer:")
print(f"  Missing values before imputing : {np.isnan(X_with_missing).sum()}")
print(f"  Missing values after  imputing : {np.isnan(X_imputed).sum()}")
print(f"  Strategy used                  : Mean")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(target_names)
print(f"\nLabel Encoder:")
print(f"  Original labels : {list(target_names)}")
print(f"  Encoded labels  : {list(y_encoded)}")

pipeline_concept = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  LogisticRegression(max_iter=1000, random_state=42))
])
pipeline_concept.fit(X_train, y_train)
pipeline_score  = pipeline_concept.score(X_test, y_test)
y_pred_pipeline = pipeline_concept.predict(X_test)
print(f"\nPipeline (Scaler + Logistic Regression):")
print(f"  Accuracy : {pipeline_score * 100:.2f}%")
print(classification_report(y_test, y_pred_pipeline, target_names=target_names))

random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)
rf_score  = random_forest.score(X_test, y_test)
y_pred_rf = random_forest.predict(X_test)
print(f"Random Forest:")
print(f"  Accuracy : {rf_score * 100:.2f}%")
print(classification_report(y_test, y_pred_rf, target_names=target_names))

param_grid = {
    "n_estimators" : [50, 100, 200],
    "max_depth"    : [None, 5, 10]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, verbose=0)
grid.fit(X_train, y_train)
grid_score = grid.score(X_test, y_test)
print(f"Grid Search CV:")
print(f"  Best Parameters : {grid.best_params_}")
print(f"  Best CV Score   : {grid.best_score_ * 100:.2f}%")
print(f"  Test Accuracy   : {grid_score * 100:.2f}%")