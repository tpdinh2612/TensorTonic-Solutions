import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    # Edge cases
    if p == 0:
        pmf = 1.0 if k == 0 else 0.0
        cdf = 1.0
        return float(pmf), float(cdf)
    
    if p == 1:
        pmf = 1.0 if k == n else 0.0
        cdf = 1.0 if k >= n else 0.0
        return float(pmf), float(cdf)

    # PMF
    pmf = comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

    # CDF
    cdf = 0.0
    for i in range(k + 1):
        cdf += comb(n, i) * (p ** i) * ((1 - p) ** (n - i))

    return float(pmf), float(cdf)