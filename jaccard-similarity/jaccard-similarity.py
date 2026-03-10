def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    set_a = set(set_a)
    set_b = set(set_b)
    union_len = len(set_a | set_b)
    
    # Return 0 if the union is empty
    return 0.0 if union_len == 0 else len(set_a & set_b) / union_len