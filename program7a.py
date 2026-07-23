import numpy as np

# Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Inputs
x1 = 0.5
x2 = 0.8

# Weights from Input to Hidden Layer
w1 = 0.2
w2 = 0.4
w3 = 0.7
w4 = 0.9

# Hidden Layer Bias
b1 = 0.1
b2 = 0.1

# Hidden Layer Output
h1 = sigmoid(x1*w1 + x2*w2 + b1)
h2 = sigmoid(x1*w3 + x2*w4 + b2)

# Weights from Hidden to Output Layer
w5 = 0.5
w6 = 0.8

# Output Bias
b3 = 0.2

# Final Output
output = sigmoid(h1*w5 + h2*w6 + b3)

print("Hidden Neuron 1 Output :", round(h1,4))
print("Hidden Neuron 2 Output :", round(h2,4))
print("Final Output           :", round(output,4))