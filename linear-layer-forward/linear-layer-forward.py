def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    # Transpose W to iterate over its columns 
    # zip(*W) unpacks the rows and regroups them into columns
    W_T = list(zip(*W))
    
    # Compute Y = XW + b
    Y = [
        [
            # Dot product of the X row and W column, plus the bias term
            sum(x_val * w_val for x_val, w_val in zip(row_X, col_W)) + b_val
            for col_W, b_val in zip(W_T, b)
        ]
        for row_X in X
    ]
    
    return Y