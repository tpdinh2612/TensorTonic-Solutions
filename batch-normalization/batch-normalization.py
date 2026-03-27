import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    x = np.asarray(x, dtype = float)
    gamma = np.asarray(gamma, dtype = float)
    beta = np.asarray(beta, dtype = float)
    
    if x.ndim == 2:
        # (N, D)
        mean = np.mean(x, axis=0, keepdims=True)
        var = np.var(x, axis=0, keepdims=True)

        x_hat = (x - mean) / np.sqrt(var + eps)

        # gamma, beta shape (D,) → broadcast ok
        out = gamma * x_hat + beta

    elif x.ndim == 4:
        # (N, C, H, W)
        mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
        var = np.var(x, axis=(0, 2, 3), keepdims=True)

        x_hat = (x - mean) / np.sqrt(var + eps)

        # reshape gamma, beta → (1, C, 1, 1)
        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)

        out = gamma * x_hat + beta

    else:
        raise ValueError("Input must be 2D or 4D")

    return out