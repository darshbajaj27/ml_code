import pandas as pd
import numpy as np

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

# Load Dataset
wine = load_wine()

X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

# ---------------- DATA CLEANING ----------------

# Simulating missing values properly using numpy's NaN
X.iloc[0:10, 2] = np.nan

# CRITICAL FIX: Split the data FIRST to prevent data leakage!
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# Handle missing values (fit on train, transform on both)
imputer = SimpleImputer(strategy='median')
X_train_clean = imputer.fit_transform(X_train)
X_test_clean = imputer.transform(X_test)

# Feature Scaling (fit on train, transform on both)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_clean)
X_test_scaled = scaler.transform(X_test_clean)

# ---------------- LDA ----------------

# Reduce dimensions to 2 (fit on train, transform on both)
lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train_scaled, y_train)
X_test_lda = lda.transform(X_test_scaled)

# ---------------- MODEL TRAINING ----------------

# Using SVM after LDA
model = SVC(
    kernel='linear',
    C=1,
    random_state=42
)

model.fit(X_train_lda, y_train)

# ---------------- PREDICTION ----------------

y_pred = model.predict(X_test_lda)

# ---------------- EVALUATION ----------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
kappa = cohen_kappa_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

print("===== LDA + SVM PERFORMANCE =====")
print(f"Accuracy          : {accuracy * 100:.2f} %")
print(f"Precision         : {precision * 100:.2f} %")
print(f"Recall            : {recall * 100:.2f} %")
print(f"F1 Score          : {f1 * 100:.2f} %")
print(f"Cohen's Kappa     : {kappa:.4f}")
print(f"Matthews Corrcoef : {mcc:.4f}")

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, y_pred))

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))

print("\n===== REDUCED DATA SHAPE =====")
print(f"Training Data Shape: {X_train_lda.shape}")
print(f"Testing Data Shape : {X_test_lda.shape}")