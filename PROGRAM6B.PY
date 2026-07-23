from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
iris = load_iris()

X = iris.data
y = iris.target

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------
# Perceptron
# ----------------------

per = Perceptron(max_iter=1000)

per.fit(X_train, y_train)

pred1 = per.predict(X_test)

acc1 = accuracy_score(y_test, pred1)

# ----------------------
# MLP
# ----------------------

mlp = MLPClassifier(hidden_layer_sizes=(10,),
                    max_iter=1000,
                    random_state=42)

mlp.fit(X_train, y_train)

pred2 = mlp.predict(X_test)

acc2 = accuracy_score(y_test, pred2)

print("Perceptron Accuracy :", round(acc1*100,2), "%")
print("MLP Accuracy        :", round(acc2*100,2), "%")