def robust_scaling(values):
    """
    Scale values using median and interquartile range.
    """
    def median(arr):
        n = len(arr)
        mid = n // 2
        if n % 2 == 0:
            return (arr[mid - 1] + arr[mid]) / 2
        else:
            return arr[mid]

    n = len(values)
    
    # case đặc biệt
    if n == 1:
        return [0.0]

    arr = sorted(values)

    # median
    med = median(arr)

    # chia lower / upper half
    mid = n // 2
    if n % 2 == 0:
        lower = arr[:mid]
        upper = arr[mid:]
    else:
        lower = arr[:mid]
        upper = arr[mid+1:]

    q1 = median(lower)
    q3 = median(upper)
    iqr = q3 - q1

    # scaling
    result = []
    for x in values: 
        if iqr == 0:
            result.append(float(x - med))
        else:
            result.append((x - med) / iqr)

    return result