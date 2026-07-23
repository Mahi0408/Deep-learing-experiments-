import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Actual parameters
actual_mean = 10
actual_std = 2

# Generate synthetic data
data = np.random.normal(actual_mean, actual_std, 1000)

# MLE Estimates
estimated_mean = np.mean(data)
estimated_variance = np.var(data)

# Display results
print("Actual Mean               :", actual_mean)
print("Estimated Mean (MLE)      :", round(estimated_mean, 4))

print("\nActual Variance           :", actual_std**2)
print("Estimated Variance (MLE)  :", round(estimated_variance, 4))

# Plot Histogram
plt.figure(figsize=(8,5))
plt.hist(data, bins=30, edgecolor='black')
plt.title("Histogram of Generated Normal Distribution")
plt.xlabel("Data Values")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()