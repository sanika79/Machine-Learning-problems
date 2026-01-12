import math

def train_linear_regression(X, y, lr=0.01, epochs=1000):
    n = len(X)        # number of samples
    m = len(X[0])     # number of features

    weights = [0] * m
    bias = 0

    for _ in range(epochs):
        total_loss = 0
        for i in range(n):
            y_pred = sum(X[i][j] * weights[j] for j in range(m)) + bias
            error = y[i] - y_pred

            # update weights and bias
            for j in range(m):
                weights[j] += lr * error * X[i][j]
            bias += lr * error

            total_loss += error**2
        # optional: print loss occasionally
        if _ % 100 == 0:
            print(f"Epoch {_}, Loss: {total_loss/n:.4f}")

    return weights, bias

def predict(X, weights, bias):
    return [sum(X[i][j] * weights[j] for j in range(len(weights))) + bias for i in range(len(X))]

# Single feature
X = [[1], [2], [3], [4]]
y = [2, 4, 6, 8]

weights, bias = train_linear_regression(X, y, lr=0.01, epochs=1000)
print("Weights:", weights)
print("Bias:", bias)

# Predictions
print("Predictions:", predict([[5], [6]], weights, bias))
