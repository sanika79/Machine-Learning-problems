import random
import math

# Distance function
def distance(a, b):
    return math.sqrt(sum((a[i] - b[i])**2 for i in range(len(a))))

# Initialize centroids
def initialize_centroids(X, k):
    return random.sample(X, k)

# Assign points to clusters
def assign_clusters(X, centroids, k):
    clusters = [[] for _ in range(k)]
    for x in X:
        idx = min(range(k), key=lambda i: distance(x, centroids[i]))
        clusters[idx].append(x)
    return clusters

# Update centroids
def update_centroids(clusters):
    new_centroids = []
    for cluster in clusters:
        mean = [sum(dim)/len(cluster) for dim in zip(*cluster)]
        new_centroids.append(mean)
    return new_centroids

# Predict function
def predict(X, centroids, k):
    return [min(range(k), key=lambda i: distance(x, centroids[i])) for x in X]


# ---------------- MAIN ---------------- #

data = [[1,2], [2,3], [3,4], [8,7], [8,8], [25,80], [24,79], [23,81]]
k = 3
max_iters = 100

centroids = initialize_centroids(data, k)

for _ in range(max_iters):
    clusters = assign_clusters(data, centroids, k)
    new_centroids = update_centroids(clusters)

    if new_centroids == centroids:
        break

    centroids = new_centroids

print("Final centroids:", centroids)
print("Clusters:", clusters)

# Predict new points
new_points = [[2,2], [9,9], [26,78]]
labels = predict(new_points, centroids, k)

print("Predictions:", labels)
