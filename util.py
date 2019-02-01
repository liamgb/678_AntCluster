import numpy as np
from scipy.spatial.distance import cosine


def similarity(gene_i, gene_j, option="cosine"):
    switcher = {
        "cosine": (lambda: 1 - cosine(gene_i, gene_j))
    }
    return switcher.get(option, 0)()


def acceptance(sim, thresh1, thresh2):
    return sim > thresh1 and sim > thresh2


def inc(M, alpha):
    return (1 - alpha) * M + alpha


def dec(M, alpha):
    return (1 - alpha) * M


if __name__ == "__main__":
    a = np.array([1, 1, 0])
    b = np.array([1, 0, 1])
    print(similarity(a, b, option="cosine"))
