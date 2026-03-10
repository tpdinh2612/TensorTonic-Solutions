def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    # Calculate the limit using the Xavier uniform formula
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    
    # Map values from [0, 1] to [-limit, limit]
    # The transformation is: scaled_val = (val * 2 * limit) - limit
    scaled_W = [
        [(val * 2 * limit) - limit for val in row]
        for row in W
    ]
    
    return scaled_W