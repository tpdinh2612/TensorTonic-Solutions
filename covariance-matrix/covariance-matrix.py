import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    x = np.asarray(X, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        return None

    # Center the Data
    μ = np.mean(x, axis=0)
    center = x - μ

    # Compute Covariance Matrix
    N = x.shape[0]
    cov = (1 / (N - 1)) * (center.T @ center)

    return cov