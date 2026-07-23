import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------------
# Linearly Separable Dataset
# -------------------------------
X1, y1 = make_classification(
    n_samples=500,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X1, y1, test_size=0.2, random_state=42
)

model = Perceptron(max_iter=1000)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Linearly Separable Accuracy:",
      round(accuracy_score(y_test, pred)*100,2), "%")

plt.figure(figsize=(6,5))
plt.scatter(X1[:,0], X1[:,1], c=y1, cmap='coolwarm')
plt.title("Linearly Separable Dataset")
plt.show()

# -------------------------------
# Non-Linearly Separable Dataset
# -------------------------------
X2, y2 = make_moons(n_samples=500, noise=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Non-Linearly Separable Accuracy:",
      round(accuracy_score(y_test, pred)*100,2), "%")

plt.figure(figsize=(6,5))
plt.scatter(X2[:,0], X2[:,1], c=y2, cmap='coolwarm')
plt.title("Non-Linearly Separable Dataset")
plt.show()