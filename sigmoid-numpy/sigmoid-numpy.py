import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    x_arr = np.asarray(x, dtype = float)
    return 1.0/(1.0 + np.exp(-x_arr))