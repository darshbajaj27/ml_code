import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Generate dataset
X, y = make_blobs(
    n_samples=300,
    centers=3,
    n_features=2,
    cluster_std=1.0,
    random_state=42
)

# Create model
model = KMeans(
    n_clusters=3,
    random_state=42
)

# Train model
model.fit(X)

# Predictions
y_pred = model.predict(X)

# Centroids
centroids = model.cluster_centers_

# Metrics
print("Inertia:", model.inertia_)

score = silhouette_score(X, y_pred)

print("Silhouette Score:", score)

# Plot clusters
plt.scatter(
    X[:,0],
    X[:,1],
    c=y_pred
)

# Plot centroids
plt.scatter(
    centroids[:,0],
    centroids[:,1],
    s=200,
    marker='X'
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.title("K-Means Clustering")

plt.show()