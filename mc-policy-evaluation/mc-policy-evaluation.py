import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    """
    Returns: V (NumPy array of shape (n_states,))
    """
   # Track the sum of returns and the number of visits for each state
    returns_sum = np.zeros(n_states)
    returns_count = np.zeros(n_states)
    
    for episode in episodes:
        # 1. Identify the first visit steps in the current episode
        first_visits = set()
        first_visit_steps = set()
        
        for t, (state, _) in enumerate(episode):
            if state not in first_visits:
                first_visits.add(state)
                first_visit_steps.add(t)
                
        # 2. Process backward to compute returns (Gt = rt + gamma * Gt+1)
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            state, reward = episode[t]
            G = reward + gamma * G
            
            # 3. If this step is a first visit, record the return
            if t in first_visit_steps:
                returns_sum[state] += G
                returns_count[state] += 1
                
    # 4. Average the returns to calculate Value (V)
    V = np.zeros(n_states)
    
    # Create a boolean mask of states that were visited at least once 
    # to avoid division by zero warnings
    visited = returns_count > 0
    V[visited] = returns_sum[visited] / returns_count[visited]
    
    return V
