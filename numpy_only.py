import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = load_diabetes(return_X_y=True)

X = np.c_[np.ones(X.shape[0]), X]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)

m = X_train.shape[0]
n = X_train.shape[1]

theta = np.zeros(n)

def cost_function(X, theta, y):
    h = np.matmul(X, theta)
    return (1 / (2 * m)) * sum((h - y) ** 2)

def R2_evaluation(X, theta, y):
    return 1 - ((sum((y - np.matmul(X, theta)) ** 2)) / (sum((y - np.mean(y)) ** 2)))

alpha = 0.08
iterations = 10000
cost_history = []

for i in range(iterations):
    h = np.matmul(X_train, theta)
    grad = (1 / m) * alpha * X_train.T.dot(h - y_train)
    theta -= grad
    cost_history.append(cost_function(X_train, theta, y_train))

print(R2_evaluation(X_test, theta, y_test))

def predict(X, theta):
    return np.matmul(X, theta)

plt.figure()
plt.plot(cost_history, [i for i in range(iterations)])
plt.figure()
plt.scatter(predict(X, theta), y)
plt.show()

    




