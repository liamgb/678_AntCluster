import numpy as np
from scipy.spatial.distance import cosine
from functools import partial


def similarity(gene_i, gene_j, *args, option=None):
    if option is None:
        option = "cosine"

    switcher = {
        "cosine": (lambda: 1 - cosine(gene_i, gene_j)),
        "discrete": (lambda: np.mean(np.equal(gene_i, gene_j))),
        "continuous": (
            lambda diff: 1 - np.mean(np.abs((gene_i - gene_j))) / diff),
        "mixed": partial(_mixed, gene_i, gene_j)
    }
    return switcher.get(option, 0)(*args)  # default return 0


def _mixed(gene_i, gene_j, mask):
    d_ind = np.where(np.isin(mask, ["discrete"]))
    c_ind = np.where(np.isin(mask, ["cosine"]))

    c_gene_i, c_gene_j = gene_i[c_ind], gene_j[c_ind]
    d_gene_i, d_gene_j = gene_i[d_ind], gene_j[d_ind]

    c_sim = similarity(c_gene_i, c_gene_j, option="cosine")
    d_sim = similarity(d_gene_i, d_gene_j, option="discrete")

    return (c_sim + d_sim) / 2


def acceptance(sim, thresh1, thresh2):
    return sim > thresh1 and sim > thresh2


def inc(M, alpha):
    return (1 - alpha) * M + alpha


def dec(M, alpha):
    return (1 - alpha) * M


def thresh(memory):
    return (np.mean(memory) + np.max(memory)) / 2.0


if __name__ == "__main__":
    a = np.array([1, 1, 0])
    b = np.array([1, 0, 1])
    print(similarity(a, b, option="cosine"))
