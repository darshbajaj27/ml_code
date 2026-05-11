# =========================
# Experiment 1: Linear Regression
# =========================

import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
    median_absolute_error
)

# =========================
# LOAD DATASET
# =========================

housing = fetch_california_housing()

X = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

y = housing.target

# =========================
# DATA CLEANING
# =========================

# Add missing values manually
X.iloc[0:20, 0] = np.nan
X.iloc[30:50, 2] = np.nan

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# HANDLE MISSING VALUES
# =========================

imputer = SimpleImputer(strategy='mean')

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# =========================
# FEATURE SCALING
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# MODEL TRAINING
# =========================

# Ridge Regression reduces overfitting
model = Ridge(alpha=1.0)

model.fit(X_train, y_train)

# =========================
# PREDICTIONS
# =========================

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# =========================
# TRAINING METRICS
# =========================

train_r2 = r2_score(
    y_train,
    y_train_pred
)

train_accuracy = train_r2 * 100

# =========================
# TESTING METRICS
# =========================

test_r2 = r2_score(
    y_test,
    y_test_pred
)

test_accuracy = test_r2 * 100

mse = mean_squared_error(
    y_test,
    y_test_pred
)

rmse = np.sqrt(mse)

mae = mean_absolute_error(
    y_test,
    y_test_pred
)

medae = median_absolute_error(
    y_test,
    y_test_pred
)

evs = explained_variance_score(
    y_test,
    y_test_pred
)

# =========================
# RESULTS
# =========================

print("\n" + "="*60)
print("           LINEAR REGRESSION PERFORMANCE")
print("="*60)

print(f"{'Training Accuracy':30s}: {train_accuracy:.2f}%")
print(f"{'Testing Accuracy':30s}: {test_accuracy:.2f}%")

print(f"{'Training R2 Score':30s}: {train_r2:.4f}")
print(f"{'Testing R2 Score':30s}: {test_r2:.4f}")

print(f"{'Mean Squared Error':30s}: {mse:.4f}")

print(f"{'Root Mean Squared Error':30s}: {rmse:.4f}")

print(f"{'Mean Absolute Error':30s}: {mae:.4f}")

print(f"{'Median Absolute Error':30s}: {medae:.4f}")

print(f"{'Explained Variance Score':30s}: {evs:.4f}")

# =========================
# OVERFITTING CHECK
# =========================

print("\n" + "="*60)
print("              OVERFITTING CHECK")
print("="*60)

difference = abs(train_r2 - test_r2)

print(f"Train-Test Difference : {difference:.4f}")

if difference < 0.05:
    print("Status                : Model is well generalized")
else:
    print("Status                : Possible overfitting detected")

# =========================
# SAMPLE PREDICTIONS
# =========================

print("\n" + "="*60)
print("              SAMPLE PREDICTIONS")
print("="*60)

for i in range(10):

    actual = y_test[i]
    predicted = y_test_pred[i]
    error = abs(actual - predicted)

    print(
        f"Sample {i+1:2d} | "
        f"Actual = {actual:.2f} | "
        f"Predicted = {predicted:.2f} | "
        f"Error = {error:.2f}"
    )

# =========================
# FEATURE IMPORTANCE
# =========================

print("\n" + "="*60)
print("              FEATURE IMPORTANCE")
print("="*60)

feature_names = housing.feature_names

for name, coef in zip(feature_names, model.coef_):

    print(f"{name:20s}: {coef:.4f}")