class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        n, m = len(X), len(X[0])  # n: samples, m: features
        self.weights = [0] * m  # Initialize weights

        for _ in range(self.epochs):
            total_loss = 0
            for i in range(n):
                y_pred = sum(X[i][j] * self.weights[j] for j in range(m)) + self.bias
                error = y[i] - y_pred

                # Update weights & bias
                for j in range(m):
                    self.weights[j] += self.learning_rate * error * X[i][j]
                self.bias += self.learning_rate * error
                total_loss += error ** 2
            
            if _ % 100 == 0:
                print(f"Epoch {_}, Loss: {total_loss/n:.4f}")

    def predict(self, X):
        return [sum(X[i][j] * self.weights[j] for j in range(len(self.weights))) + self.bias for i in range(len(X))]
