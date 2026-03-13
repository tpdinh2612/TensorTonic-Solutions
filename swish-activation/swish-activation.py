import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    x = np.asarray(x, dtype = float)
    #Clip x to prevent overflow in np.exp(-x)
    x_clipped = np.clip(x, -500, 500)
    sigmoid = 1 / (1 + np.exp(-x))
    return x*sigmoid