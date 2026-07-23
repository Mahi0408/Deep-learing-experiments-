import numpy as np
import matplotlib.pyplot as plt

# Input Values
x = np.linspace(-10, 10, 100)

# Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ReLU Activation Function
def relu(x):
    return np.maximum(0, x)

# Calculate Outputs
sigmoid_output = sigmoid(x)
relu_output = relu(x)

# Print Sample Outputs
print("Input\tSigmoid\t\tReLU")
for i in range(0, len(x), 10):
    print(f"{x[i]:.2f}\t{sigmoid_output[i]:.4f}\t\t{relu_output[i]:.2f}")

# Plot Graph
plt.figure(figsize=(10,5))

plt.plot(x, sigmoid_output, label='Sigmoid', linewidth=2)
plt.plot(x, relu_output, label='ReLU', linewidth=2)

plt.title("Comparison of Sigmoid and ReLU Activation Functions")
plt.xlabel("Input Values")
plt.ylabel("Output")
plt.legend()
plt.grid(True)

plt.show()