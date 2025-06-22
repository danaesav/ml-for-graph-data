import numpy as np


def jaccard_similarity(set1, set2):
    intersection = np.logical_and(set1, set2).sum()
    union = np.logical_or(set1, set2).sum()
    if union == 0:
        return 1.0
    return float(intersection) / float(union)
