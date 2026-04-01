import math

def ndcg(relevance_scores, k):
    """
    Compute NDCG@k.
    """
    def dcg(scores):
        res = 0.0
        for i, rel in enumerate(scores):
            gain = (2 ** rel) - 1
            discount = math.log2(i + 2)  # i=0 -> log2(2)=1
            res += gain / discount
        return res

    # lấy top k
    actual = relevance_scores[:k]
    ideal = sorted(relevance_scores, reverse=True)[:k]

    dcg_val = dcg(actual)
    idcg_val = dcg(ideal)

    if idcg_val == 0:
        return 0.0

    return dcg_val / idcg_val