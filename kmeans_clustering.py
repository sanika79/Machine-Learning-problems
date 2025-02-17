import random

class KMeans:
    def __init__(self, k, max_iters=100):
        self.k = k  # Number of clusters
        self.max_iters = max_iters  # Maximum iterations
        self.centroids = []  # Stores cluster centroids

    def initialize_centroids(self, X):
        """Randomly initialize k centroids from the dataset."""
        self.centroids = random.sample(X, self.k)
        ### k centroids have been inilialized

    def euclidean_distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        return sum((point1[i] - point2[i]) ** 2 for i in range(len(point1))) ** 0.5

    def assign_clusters(self, X):
        """Assign each data point to the nearest centroid."""
        clusters = [[] for _ in range(self.k)]
        ## iterate through the data
        for x in X:
            distances = [self.euclidean_distance(x, centroid) for centroid in self.centroids]
            ### data = [p1, p2, p3.....]
            ### find distance of each data point to all k centroids
            ### if point p1 is closest to c3, then cluster index = 2 (3rd cluster)
            cluster_index = distances.index(min(distances))    
            clusters[cluster_index].append(x)       ## len of clusters = number of clusters = k
            ## clisters = [[cluster 0], [cluster1]......[cluster[k]]]  for k clusters
        return clusters

    def update_centroids(self, clusters):
        """Update centroids as the mean of assigned points."""
        new_centroids = []
        for cluster in clusters:
            if cluster:  # Avoid empty clusters
                new_centroid = [sum(dim) / len(cluster) for dim in zip(*cluster)]
            else:  
                new_centroid = random.choice(clusters)  # Reinitialize empty cluster
            new_centroids.append(new_centroid)
        return new_centroids

    def has_converged(self, old_centroids, new_centroids):
        """Check if centroids have converged (no significant change)."""
        return all(self.euclidean_distance(old, new) < 1e-6 for old, new in zip(old_centroids, new_centroids))

    def fit(self, X):
        """Main function to run K-Means."""
        self.initialize_centroids(X)

        for _ in range(self.max_iters):
            clusters = self.assign_clusters(X)
            
            new_centroids = self.update_centroids(clusters)
            
            if self.has_converged(self.centroids, new_centroids):
                break
            
            self.centroids = new_centroids  # Update centroids
        
        return self.centroids, clusters

    def predict(self, X):
        """Predict the cluster for each input point."""
        return [min(range(self.k), key=lambda i: self.euclidean_distance(x, self.centroids[i])) for x in X]
    


            # Sample dataset (each point is a 2D coordinate)
data = [
    [1, 2], [2, 3], [3, 4], [8, 7], [8, 8], [25, 80], [24, 79], [23, 81]
]

# Number of clusters
k = 3

# Initialize and run K-Means
kmeans = KMeans(k=k)
centroids, clusters = kmeans.fit(data)

# Output results
print("Final centroids:")
for i, centroid in enumerate(centroids):
    print(f"Cluster {i}: {centroid}")

print("\nCluster assignments:")
for i, cluster in enumerate(clusters):
    print(f"Cluster {i}: {cluster}")

# Predict the cluster for new points
new_points = [[2, 2], [9, 9], [26, 78]]
predictions = kmeans.predict(new_points)
print("\nPredicted clusters for new points:")
for point, cluster in zip(new_points, predictions):
    print(f"Point {point} -> Cluster {cluster}")

