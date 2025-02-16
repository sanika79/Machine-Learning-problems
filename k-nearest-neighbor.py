class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def distance(self, x1, x2):
        return sum((x1[i] - x2[i]) ** 2 for i in range(len(x1))) ** 0.5

    def predict(self, X):
        predictions = []
        for x in X:
            distances = [(self.distance(x, self.X_train[i]), self.y_train[i]) for i in range(len(self.X_train))]
            distances.sort()
            neighbors = [distances[i][1] for i in range(self.k)]
            predictions.append(max(set(neighbors), key=neighbors.count))
        return predictions
