import numpy as np

# Initial Weight
weight = 0.5

# Bias
bias = 0.2

# Input
x = 1

# Actual Output
y = 1

# Learning Rate
learning_rate = 0.1

# Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Forward Pass
z = weight * x + bias
prediction = sigmoid(z)

# Error
error = prediction - y

# Gradient Calculation
gradient = error * prediction * (1 - prediction) * x

# Weight Update
new_weight = weight - learning_rate * gradient

print("Initial Weight :", round(weight,4))
print("Prediction     :", round(prediction,4))
print("Error          :", round(error,4))
print("Gradient       :", round(gradient,4))
print("Updated Weight :", round(new_weight,4))