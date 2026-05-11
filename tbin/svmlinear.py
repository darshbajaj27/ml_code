# =========================
# Experiment 3: Linear SVM
# =========================

import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef
)

# =========================
# LOAD DATASET
# =========================

cancer = load_breast_cancer()

X = pd.DataFrame(
    cancer.data,
    columns=cancer.feature_names
)

y = cancer.target

# =========================
# DATA CLEANING
# =========================

# Add missing values manually
X.iloc[0:15, 2] = np.nan
X.iloc[20:30, 5] = np.nan

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
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

# Linear kernel works well for linearly separable data
# C controls overfitting

model = SVC(
    kernel='linear',
    C=1,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# TRAIN PREDICTIONS
# =========================

y_train_pred = model.predict(X_train)

# =========================
# TEST PREDICTIONS
# =========================

y_test_pred = model.predict(X_test)

# =========================
# TRAINING METRICS
# =========================

train_accuracy = accuracy_score(
    y_train,
    y_train_pred
)

# =========================
# TESTING METRICS
# =========================

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

precision = precision_score(
    y_test,
    y_test_pred
)

recall = recall_score(
    y_test,
    y_test_pred
)

f1 = f1_score(
    y_test,
    y_test_pred
)

kappa = cohen_kappa_score(
    y_test,
    y_test_pred
)

mcc = matthews_corrcoef(
    y_test,
    y_test_pred
)

# =========================
# RESULTS
# =========================

print("\n" + "="*60)
print("               LINEAR SVM PERFORMANCE")
print("="*60)

print(f"{'Training Accuracy':30s}: {train_accuracy*100:.2f}%")
print(f"{'Testing Accuracy':30s}: {test_accuracy*100:.2f}%")

print(f"{'Precision':30s}: {precision*100:.2f}%")
print(f"{'Recall':30s}: {recall*100:.2f}%")
print(f"{'F1 Score':30s}: {f1*100:.2f}%")

print(f"{'Cohen Kappa Score':30s}: {kappa:.4f}")
print(f"{'Matthews Corrcoef':30s}: {mcc:.4f}")

# =========================
# OVERFITTING CHECK
# =========================

print("\n" + "="*60)
print("              OVERFITTING CHECK")
print("="*60)

difference = abs(train_accuracy - test_accuracy)

print(f"Train-Test Difference : {difference:.4f}")

if difference < 0.05:
    print("Status                : Model is well generalized")
else:
    print("Status                : Possible overfitting detected")

# =========================
# CONFUSION MATRIX
# =========================

print("\n" + "="*60)
print("              CONFUSION MATRIX")
print("="*60)

print(confusion_matrix(y_test, y_test_pred))

# =========================
# CLASSIFICATION REPORT
# =========================

print("\n" + "="*60)
print("           CLASSIFICATION REPORT")
print("="*60)

print(classification_report(y_test, y_test_pred))

# =========================
# SAMPLE PREDICTIONS
# =========================

print("\n" + "="*60)
print("            SAMPLE PREDICTIONS")
print("="*60)

for i in range(10):

    print(
        f"Actual: {y_test[i]} | "
        f"Predicted: {y_test_pred[i]}"
    )