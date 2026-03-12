def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    if not points:
        return []

    # Get dimensionality of the points (e.g., 2D, 3D)
    dim = len(points[0])
    
    # Initialize sums for each cluster with zero vectors and a count array
    sums = [[0.0] * dim for _ in range(k)]
    counts = [0] * k

    # Accumulate the sum of points for each cluster
    for i in range(len(points)):
        cluster_idx = assignments[i]
        counts[cluster_idx] += 1
        for d in range(dim):
            sums[cluster_idx][d] += points[i][d]

    # Compute the mean (sum / count)
    centroids = []
    for i in range(k):
        if counts[i] == 0:
            # Handle empty clusters as per requirements
            centroids.append([0.0] * dim)
        else:
            # Calculate the average for each dimension
            mean_vector = [sums[i][d] / counts[i] for d in range(dim)]
            centroids.append(mean_vector)

    return centroids
        