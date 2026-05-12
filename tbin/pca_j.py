import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Generate dataset
X, y = make_classification(
    n_samples=500,
    n_features=10,
    n_informative=8,
    n_redundant=1,
    n_classes=3,
    n_clusters_per_class=1,
    class_sep=3,
    random_state=42
)

print("Original Shape:", X.shape)

# Scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("Reduced Shape:", X_pca.shape)

print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

print(
    "Total Variance Retained:",
    pca.explained_variance_ratio_.sum()
)

# Plot
plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=y
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title("PCA Dimensionality Reduction")

plt.show()