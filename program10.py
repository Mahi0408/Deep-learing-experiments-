import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Different feature dimensions
dimensions = [2, 10, 50, 100]

accuracies = []

for dim in dimensions:

    # Generate Dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=dim,
        n_informative=2,
        n_redundant=0,
        random_state=42
    )

    # Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Train KNN Classifier
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    print(f"Features: {dim} --> Accuracy: {acc*100:.2f}%")

# Plot Accuracy vs Dimensions
plt.figure(figsize=(8,5))
plt.plot(dimensions, accuracies, marker='o', linewidth=2)

plt.title("Effect of Curse of Dimensionality")
plt.xlabel("Number of Features")
plt.ylabel("Accuracy")

plt.grid(True)

plt.show()