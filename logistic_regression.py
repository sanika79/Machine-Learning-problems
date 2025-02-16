class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def sigmoid(self, z):
        return 1 / (1 + (2.71828 ** -z))

    def fit(self, X, y):
        n, m = len(X), len(X[0])
        self.weights = [0] * m

        for _ in range(self.epochs):
            total_loss = 0
            for i in range(n):
                z = sum(X[i][j] * self.weights[j] for j in range(m)) + self.bias
                y_pred = self.sigmoid(z)
                error = y[i] - y_pred

                # Update weights & bias
                for j in range(m):
                    self.weights[j] += self.learning_rate * error * X[i][j]
                self.bias += self.learning_rate * error
                total_loss += -(y[i] * z - (1 - y[i]) * (1 - z))

            if _ % 100 == 0:
                print(f"Epoch {_}, Loss: {total_loss:.4f}")

    def predict(self, X):
        return [1 if self.sigmoid(sum(X[i][j] * self.weights[j] for j in range(len(self.weights))) + self.bias) > 0.5 else 0 for i in range(len(X))]
