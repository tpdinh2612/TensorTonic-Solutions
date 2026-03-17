import numpy as np

def nadam_step(w, m, v, grad, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Perform one Nadam update step.
    """
    w_arr = np.array(w)
    m_arr = np.array(m)
    v_arr = np.array(v)
    grad_arr = np.array(grad)
    
    # 1. Update the first moment (momentum)
    m_new = beta1 * m_arr + (1.0 - beta1) * grad_arr
    
    # 2. Update the second moment (velocity)
    v_new = beta2 * v_arr + (1.0 - beta2) * (grad_arr ** 2)
    
    # 3. Apply the Nesterov adjustment to the momentum
    m_hat = beta1 * m_new + (1.0 - beta1) * grad_arr
    
    # 4. Update the weights/parameters
    w_new = w_arr - lr * m_hat / (np.sqrt(v_new) + eps)
    
    # Return as standard Python lists if the inputs were lists, otherwise return as arrays
    if isinstance(w, list):
        return w_new.tolist(), m_new.tolist(), v_new.tolist()
    
    return w_new, m_new, v_new