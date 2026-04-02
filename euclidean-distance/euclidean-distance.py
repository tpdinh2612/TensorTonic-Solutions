import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    x = np.asarray(x, dtype = float)
    y = np.asarray(y, dtype = float)
    l2Distance = np.sqrt(np.sum((x - y) ** 2))
    return float(l2Distance)