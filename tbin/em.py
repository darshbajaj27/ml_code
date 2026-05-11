# =========================
# Experiment: EM Algorithm using Gaussian Mixture Model (GMM)
# =========================

import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

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

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target

# =========================
# DATA CLEANING
# =========================

# Add missing values manually
X.iloc[0:10, 1] = np.nan
X.iloc[20:25, 2] = np.nan

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
# MODEL TRAINING
# =========================

# Regularization added using reg_covar
# Helps prevent overfitting

model = GaussianMixture(
    n_components=3,
    covariance_type='diag',
    reg_covar=1e-3,
    n_init=5,
    random_state=42
)

# Train model
train_clusters = model.fit_predict(X_train)

# =========================
# CLUSTER TO LABEL MAPPING
# =========================

cluster_mapping = {}

for cluster in range(3):

    labels = y_train[train_clusters == cluster]

    majority_label = pd.Series(labels).mode()[0]

    cluster_mapping[cluster] = majority_label

# =========================
# TRAIN PREDICTIONS
# =========================

train_pred_clusters = model.predict(X_train)

y_train_pred = np.vectorize(
    cluster_mapping.get
)(train_pred_clusters)

# =========================
# TEST PREDICTIONS
# =========================

test_pred_clusters = model.predict(X_test)

y_test_pred = np.vectorize(
    cluster_mapping.get
)(test_pred_clusters)

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
print("         EM ALGORITHM (GMM) PERFORMANCE")
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
# CLUSTER MAPPING
# =========================

print("\n" + "="*60)
print("             CLUSTER MAPPING")
print("="*60)

for cluster, label in cluster_mapping.items():

    print(
        f"Cluster {cluster}  --->  Iris Class {label}"
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

# =========================
# CLUSTER MEANS
# =========================

print("\n" + "="*60)
print("         GMM CLUSTER MEANS")
print("="*60)

print(model.means_)