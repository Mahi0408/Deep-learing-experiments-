import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

# Generate Dataset
X, y = make_regression(n_samples=200, n_features=1, noise=20, random_state=42)

# Normalize Features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize Parameters
w = 0
b = 0
learning_rate = 0.1
epochs = 100

m = len(y)
loss_history = []

# Gradient Descent
for epoch in range(epochs):
    y_pred = w * X.flatten() + b

    dw = (-2/m) * np.sum(X.flatten() * (y - y_pred))
    db = (-2/m) * np.sum(y - y_pred)

    w -= learning_rate * dw
    b -= learning_rate * db

    loss = np.mean((y - y_pred) ** 2)
    loss_history.append(loss)

print("Final Weight :", round(w,4))
print("Final Bias   :", round(b,4))
print("Final Loss   :", round(loss_history[-1],4))

# Plot Learning Curve
plt.figure(figsize=(8,5))
plt.plot(loss_history, linewidth=2)
plt.title("Gradient Descent Learning Curve")
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.show()