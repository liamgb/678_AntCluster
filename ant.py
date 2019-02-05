import numpy as np
from util import thresh


class Ant(object):
    def __init__(self, gene, true_label=None):
        self.true_label = true_label
        self.gene = np.array(gene)  # np.array
        self.memory = []  # list of similarities with previously met ants
        self.label = 0
        self.M = 0
        self.M_p = 0
        self.treshold = 0

    def update_treshold(self):
        self.treshold = thresh(self.memory)

    def exile(self):
        self.label = 0
        self.M = 0
        self.M_p = 0

    def turnover(self, new_nest):
        self.label = new_nest


if __name__ == "__main__":
    ant1 = Ant([1, 1, 0])
