# =========================
# Experiment: PCA + SVM
# =========================

import pandas as pd
import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

digits = load_digits()

X = pd.DataFrame(digits.data)
y = digits.target

# =========================
# DATA CLEANING
# =========================

# Add missing values manually
X.iloc[0:20, 7] = np.nan
X.iloc[40:60, 15] = np.nan

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

imputer = SimpleImputer(strategy='median')

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# =========================
# FEATURE SCALING
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# PCA DIMENSION REDUCTION
# =========================

# Keep 95% variance
pca = PCA(
    n_components=0.95,
    random_state=42
)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# =========================
# MODEL TRAINING
# =========================

model = SVC(
    kernel='rbf',
    C=1,
    random_state=42
)

model.fit(X_train_pca, y_train)

# =========================
# TRAIN PREDICTIONS
# =========================

y_train_pred = model.predict(X_train_pca)

# =========================
# TEST PREDICTIONS
# =========================

y_test_pred = model.predict(X_test_pca)

# =========================
# TRAINING ACCURACY
# =========================

train_accuracy = accuracy_score(
    y_train,
    y_train_pred
)

# =========================
# TESTING ACCURACY
# =========================

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

precision = precision_score(
    y_test,
    y_test_pred,
    average='weighted'
)

recall = recall_score(
    y_test,
    y_test_pred,
    average='weighted'
)

f1 = f1_score(
    y_test,
    y_test_pred,
    average='weighted'
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
print("              PCA + SVM PERFORMANCE")
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
# PCA RESULTS
# =========================

print("\n" + "="*60)
print("                  PCA RESULTS")
print("="*60)

print(f"Original Shape (Train) : {X_train.shape}")

print(f"Reduced Shape (Train)  : {X_train_pca.shape}")

print(f"Number of Components   : {pca.n_components_}")

print(
    f"Total Variance Kept    : "
    f"{sum(pca.explained_variance_ratio_) * 100:.2f}%"
)

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