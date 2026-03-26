import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    
    # reshape nếu là 1D
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
    
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    
    # broadcasting để tính khoảng cách
    diff = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)  # shape (n_test, n_train)
    
    # sort indices theo khoảng cách
    sorted_idx = np.argsort(dist, axis=1)
    
    # lấy k nearest
    k_eff = min(k, n_train)
    result = sorted_idx[:, :k_eff]
    
    # pad nếu k > n_train
    if k > n_train:
        pad = -1 * np.ones((n_test, k - n_train), dtype=int)
        result = np.hstack([result, pad])
    
    return result.astype(int)