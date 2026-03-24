import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    # Write code here
    p = np.array(p)
    y = np.array(y)
    
    p = np.clip(p, 1e-15, 1 - 1e-15)
    
    term1 = y * ((1 - p) ** gamma) * np.log(p)
    term2 = (1 - y) * (p ** gamma) * np.log(1 - p)
    
    loss = -(term1 + term2)
    
    return np.mean(loss)