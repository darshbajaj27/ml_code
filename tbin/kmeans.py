# =========================
# Experiment 7: K-Means Clustering
# =========================

import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    accuracy_score
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

# Add missing values
X.iloc[0:10, 2] = np.nan
X.iloc[20:25, 1] = np.nan

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

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

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
# EVALUATION
# =========================

# Clustering Metrics
silhouette = silhouette_score(
    X_train,
    train_clusters
)

train_ari = adjusted_rand_score(
    y_train,
    train_pred_clusters
)

test_ari = adjusted_rand_score(
    y_test,
    test_pred_clusters
)

# Accuracy Metrics
train_accuracy = accuracy_score(
    y_train,
    y_train_pred
)

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

# =========================
# RESULTS
# =========================

print("\n" + "="*60)
print("           K-MEANS CLUSTERING PERFORMANCE")
print("="*60)

print(f"{'Training Accuracy':30s}: {train_accuracy*100:.2f}%")
print(f"{'Testing Accuracy':30s}: {test_accuracy*100:.2f}%")

print(f"{'Training ARI Score':30s}: {train_ari:.4f}")
print(f"{'Testing ARI Score':30s}: {test_ari:.4f}")

print(f"{'Silhouette Score':30s}: {silhouette:.4f}")

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
print("              CLUSTER MAPPING")
print("="*60)

for cluster, label in cluster_mapping.items():

    print(
        f"Cluster {cluster}  --->  Iris Class {label}"
    )

# =========================
# CLUSTER CENTERS
# =========================

print("\n" + "="*60)
print("               CLUSTER CENTERS")
print("="*60)

print(model.cluster_centers_)

# =========================
# SAMPLE CLUSTER LABELS
# =========================

print("\n" + "="*60)
print("          FIRST 20 CLUSTER LABELS")
print("="*60)

print(test_pred_clusters[:20])

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