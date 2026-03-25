import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)
    
    # Compute norms (magnitudes)
    norm_v = np.linalg.norm(v)
    norm_w = np.linalg.norm(w)
    
    # Handle zero vectors
    if norm_v < 1e-10 or norm_w < 1e-10:
        return np.nan
    
    # Compute cosine of angle using dot product
    cos_theta = np.dot(v, w) / (norm_v * norm_w)
    
    # Clamp to avoid numerical errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Return angle in radians
    return np.arccos(cos_theta)