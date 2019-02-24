from ant import Ant
from util import *
from random import sample
from collections import defaultdict
from tqdm import tqdm


class AntCluster(object):
    def __init__(self, data, alpha=0.2, label=None, mask=None, option=None):
        self.mask = mask
        self.option = option
        self.al = alpha
        self.nest_counter = 1
        if label is None:
            self.ants = set([Ant(datum) for datum in data.tolist()])
        else:
            self.ants = set([Ant(datum, true_label=label[i])
                             for i, datum in enumerate(data.tolist())])

    def group(self):
        groups = defaultdict(list)

        for ant in self.ants:
            groups[ant.label].append(ant)

        return groups.values()

    def similate(self, runs=None, burns=None):
        self.statistic = []

        if runs is None:
            runs = len(self.ants) * 100
        if burns is None:
            burns = int(runs / 10)

        # initial burn in
        for _ in tqdm(range(burns)):
            ant1, ant2 = sample(self.ants, 2)
            self.meet(ant1, ant2)

        # colonizing
        for _ in tqdm(range(runs)):
            ant1, ant2 = sample(self.ants, 2)
            self.meet(ant1, ant2, action=self.colonize)

    def meet(self, ant1, ant2, action=lambda *args: None):
        if self.option == "mixed":
            sim = similarity(
                ant1.gene, ant2.gene, self.mask, option=self.option)
        else:
            sim = similarity(ant1.gene, ant2.gene, option=self.option)

        action(ant1, ant2, sim)

        ant1.memory.append(sim)
        ant1.update_treshold()
        ant2.memory.append(sim)
        ant2.update_treshold()

    def colonize(self, ant1, ant2, sim):
        accept = acceptance(sim, ant1.treshold, ant2.treshold)

        if ant1.label == ant2.label == 0 and accept:
            # create new nest
            ant1.label = self.nest_counter
            ant2.label = self.nest_counter
            self.nest_counter += 1
            self.statistic.append([1, 0, 0, 0, 0])

        elif ant1.label == 0 and ant2.label != 0 and accept:
            # join nest
            ant1.label = ant2.label

            self.statistic.append([0, 1, 0, 0, 0])
        elif ant1.label != 0 and ant2.label == 0 and accept:
            ant2.label = ant1.label

            self.statistic.append([0, 1, 0, 0, 0])

        elif ant1.label == ant2.label != 0 and accept:
            # positive meeting, increase M and M_p
            ant1.M, ant2.M = inc(ant1.M, self.al), inc(ant2.M, self.al)
            ant1.M_p, ant2.M_p = inc(ant1.M_p, self.al), inc(ant2.M_p, self.al)

            self.statistic.append([0, 0, 1, 0, 0])

        elif ant1.label == ant2.label != 0 and not accept:
            # negative meeting, increase M, decrease M_p
            ant1.M, ant2.M = inc(ant1.M, self.al), inc(ant2.M, self.al)
            ant1.M_p, ant2.M_p = dec(ant1.M_p, self.al), dec(ant2.M_p, self.al)

            # kick the less confident ant out
            if ant1.M_p > ant2.M_p:
                ant2.exile()
            elif ant1.M_p < ant2.M_p:
                ant1.exile()

            self.statistic.append([0, 0, 0, 1, 0])

        elif ant1.label != ant2.label and accept:
            # diff nest meeting, ant convertion
            if ant1.M > ant2.M:
                ant2.turnover(ant1.label)
            elif ant1.M < ant2.M:
                ant1.turnover(ant2.label)

            self.statistic.append([0, 0, 0, 0, 1])

        else:
            pass


if __name__ == "__main__":
    pass
