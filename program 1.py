import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

# Load Dataset
data = load_diabetes()
X = data.data[:, 0].reshape(-1, 1)  # Use one feature
y = data.target

# Normalize the feature
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize parameters
m = len(y)
w = 0
b = 0
learning_rate = 0.1
epochs = 200

loss_history = []

# Gradient Descent
for i in range(epochs):
    y_pred = w * X.flatten() + b

    dw = (-2 / m) * np.sum(X.flatten() * (y - y_pred))
    db = (-2 / m) * np.sum(y - y_pred)

    w = w - learning_rate * dw
    b = b - learning_rate * db

    loss = np.mean((y - y_pred) ** 2)
    loss_history.append(loss)

print("Final Weight :", round(w, 4))
print("Final Bias   :", round(b, 4))
print("Final Loss   :", round(loss_history[-1], 4))

# Plot Learning Curve
plt.figure(figsize=(8,5))
plt.plot(loss_history, color='blue')
plt.title("Learning Curve using Gradient Descent")
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.show()