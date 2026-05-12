import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate synthetic dataset
X, y = make_regression(
    n_samples=100,
    n_features=1,
    noise=10,
    random_state=42
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Slope:", model.coef_)
print("Intercept:", model.intercept_)
print("RMSE:", rmse)

# Plot
plt.scatter(X_test, y_test)

sorted_indices = np.argsort(X_test[:,0])

plt.plot(
    X_test[sorted_indices],
    y_pred[sorted_indices]
)

plt.xlabel("Feature")
plt.ylabel("Target")

plt.title("Linear Regression")

plt.show()