import math

def distance(a, b):
    return math.sqrt(sum((a[i] - b[i])**2 for i in range(len(a))))

def knn_predict(X_train, y_train, X_test, k):
    predictions = []

    for x in X_test:
        # Compute distances to all training points
        dists = [(distance(x, X_train[i]), y_train[i]) 
                 for i in range(len(X_train))]

        # Sort by distance
        dists.sort()

        # Take k nearest labels
        labels = [dists[i][1] for i in range(k)]

        # Majority vote
        predictions.append(max(set(labels), key=labels.count))

    return predictions
